// hellofem::app — physics field solvers (electrostatics / heat / solid)
// SPDX-License-Identifier: MIT

#include "physics.h"

#include "basis/element-families.h"
#include "fem/CoordinateElement.h"
#include "fem/assembler.h"
#include "fem/dofmapbuilder.h"
#include "fem/facet_precompute.h"
#include "fem/precompute.h"
#include "fem/sparsitybuild.h"
#include "kernels.h"
#include "la/Vector.h"
#include "mesh/Topology.h"
#include "mesh/cell_types.h"
#include "mesh/utils.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace hellofem::app {
    namespace {

        using B = basis::element::family;
        using LV = basis::element::lagrange_variant;
        using DV = basis::element::dpc_variant;

        /// Build a Lagrange FunctionSpace of the given order and value dim
        /// (1 = scalar, 3 = vector displacement).
        std::shared_ptr<fem::FunctionSpace<double>> make_space(
            std::shared_ptr<const mesh::Mesh<double>> mesh, int order, int value_dim)
        {
            auto cell_type = mesh->topology()->cell_type();
            auto ctype = mesh::cell_type_to_basix_type(cell_type);
            auto fe = std::make_shared<fem::FiniteElement<double>>(
                basis::create_element<double>(B::P, ctype, order, LV::equispaced,
                    DV::unset, false),
                value_dim == 1
                    ? std::nullopt
                    : std::optional<std::vector<std::size_t>> {
                          std::vector<std::size_t> {
                              static_cast<std::size_t>(value_dim)}});
            // The dofmap is built from the ELEMENT's own dof layout so a
            // blocked (vector) element yields a bs=3 DofMap.
            auto layout = fe->create_dof_layout();
            auto [imap, bs, dofmaps] = fem::build_dofmap_data(*mesh->topology(),
                {layout}, nullptr);
            auto dmap = std::make_shared<fem::DofMap>(layout,
                std::make_shared<common::IndexMap>(std::move(imap)), bs,
                std::move(dofmaps.front()), bs);
            return std::make_shared<fem::FunctionSpace<double>>(mesh, fe, dmap);
        }

        /// Zero the columns of the dofs marked in `bcmask` (indexed by
        /// physical dof, as `DirichletBC::mark_dofs` writes it). The matrix
        /// was assembled with the Dirichlet rows zeroed but the columns
        /// kept, so that the known boundary values could be moved to the
        /// right-hand side first.
        void zero_marked_columns(la::MatrixCSR<double>& A,
            std::span<const std::int8_t> bcmask)
        {
            const auto [bs0, bs1] = A.block_size();
            auto& vals = A.values();
            const auto& cols = A.cols();
            for (std::size_t j = 0; j < cols.size(); ++j) {
                const std::size_t c = static_cast<std::size_t>(cols[j]);
                double* block = &vals[j * static_cast<std::size_t>(bs0 * bs1)];
                for (int i0 = 0; i0 < bs0; ++i0)
                    for (int i1 = 0; i1 < bs1; ++i1)
                        if (bcmask[static_cast<std::size_t>(bs1) * c
                                + static_cast<std::size_t>(i1)])
                            block[static_cast<std::size_t>(i0 * bs1 + i1)] = 0.0;
            }
        }

        /// Impose the Dirichlet data on an assembled system: subtract the
        /// known boundary-column contribution from the right-hand side,
        /// zero those columns, set the diagonal and write the values, so
        /// the boundary rows read `u = g`. `marked` is the dof marker of
        /// `bcs`.
        void impose_dirichlet(la::MatrixCSR<double>& A, la::Vector<double>& b,
            const fem::Form<double>& a,
            const std::vector<fem::DirichletBC<double>>& bcs,
            std::span<const std::int8_t> marked)
        {
            if (bcs.empty())
                return;
            std::vector<std::reference_wrapper<const fem::DirichletBC<double>>> bref(
                bcs.begin(), bcs.end());
            const auto& dofmap = *a.function_spaces()[0]->dofmap();
            const int bs = dofmap.index_map_bs();

            // b -= A[:, bc] * g.
            la::Vector<double> g(dofmap.index_map, bs);
            g.set(0.0);
            fem::set_bc(std::span(g.array()), bref,
                std::optional<std::span<const double>> {}, 1.0);
            la::Vector<double> lift(dofmap.index_map, bs);
            lift.set(0.0);
            A.mult(g, lift);
            for (std::size_t i = 0; i < b.array().size(); ++i)
                b.array()[i] -= lift.array()[i];

            zero_marked_columns(A, marked);
            if (bs == 1)
                fem::set_diagonal(A.mat_set_values<1, 1>(), a, bref, 1.0);
            else
                fem::set_diagonal(A.mat_set_values<3, 3>(), a, bref, 1.0);
            fem::set_bc(std::span(b.array()), bref,
                std::optional<std::span<const double>> {}, 1.0);
        }

        /// Write the Dirichlet values of `bcs` into `x`.
        void impose_values(std::span<double> x,
            const std::vector<fem::DirichletBC<double>>& bcs)
        {
            std::vector<std::reference_wrapper<const fem::DirichletBC<double>>> bref(
                bcs.begin(), bcs.end());
            fem::set_bc(x, bref, std::optional<std::span<const double>> {}, 1.0);
        }

        /// One integral over the given cells or facets, with the given kernel
        /// and the coefficient indices the kernel reads.
        using Integrals = std::map<std::pair<fem::IntegralType, int>,
            std::vector<fem::Form<double>::integral_data>>;

        /// Whether a property reads the solution, i.e. whether the physics
        /// it belongs to has to be linearized and iterated.
        bool solution_dependent(const std::shared_ptr<CellProperty>& property)
        {
            return property and property->field_dependent();
        }

        /// Facet indices carrying the given 1-based boundary ids.
        std::vector<std::int32_t> tagged_facets(
            const mesh::MeshTags<int>& tags, const std::set<int>& ids)
        {
            std::vector<std::int32_t> facets;
            for (std::size_t i = 0; i < tags.indices().size(); ++i)
                if (ids.contains(tags.values()[i]))
                    facets.push_back(tags.indices()[i]);
            return facets;
        }

        /// Physical coefficient elements of a form, in order.
        std::vector<const fem::FiniteElement<double>*> coefficient_elements(
            const std::vector<std::shared_ptr<const fem::Function<double>>>& coeffs)
        {
            std::vector<const fem::FiniteElement<double>*> elements;
            elements.reserve(coeffs.size());
            for (const auto& c : coeffs)
                elements.push_back(c->function_space()->element().get());
            return elements;
        }

        /// Indices of the coefficients of a form: 0, 1, ... in order.
        std::vector<int> coefficient_indices(std::size_t count)
        {
            std::vector<int> indices(count);
            std::iota(indices.begin(), indices.end(), 0);
            return indices;
        }

        /// `make_cell_kernel` reads the precomputed data by reference, so the
        /// kernel has to keep it alive next to itself.
        fem::kernel_t<double> keeping_cell_kernel(
            std::shared_ptr<const fem::PrecomputeData<double>> pre,
            fem::cell_kernel_weak_fn_t<double> w)
        {
            auto kernel = fem::make_cell_kernel(*pre, w);
            return [pre = std::move(pre),
                       kernel = std::move(kernel)](double* Ae, const double* coeffs,
                       const double* constants, const double* cds, const int* perm,
                       const std::uint8_t* entity, void* ctx) mutable {
                kernel(Ae, coeffs, constants, cds, perm, entity, ctx);
            };
        }

        /// Facet counterpart of `keeping_cell_kernel`.
        fem::kernel_t<double> keeping_facet_kernel(
            std::shared_ptr<const fem::FacetPrecomputeData<double>> pre,
            fem::facet_kernel_weak_fn_t<double> w)
        {
            auto kernel = fem::make_facet_kernel(*pre, w);
            return [pre = std::move(pre),
                       kernel = std::move(kernel)](double* Ae, const double* coeffs,
                       const double* constants, const double* cds, const int* perm,
                       const std::uint8_t* entity, void* ctx) mutable {
                kernel(Ae, coeffs, constants, cds, perm, entity, ctx);
            };
        }

        /// Integrals of one cell or exterior-facet block of the given kernel.
        Integrals single_integral(fem::IntegralType type,
            const std::vector<std::int32_t>& entities, fem::kernel_t<double> kernel,
            std::size_t num_coeffs)
        {
            Integrals integrals;
            integrals[{type, 0}] = {
                {std::move(kernel), entities, coefficient_indices(num_coeffs)}};
            return integrals;
        }

    } // namespace

    // ---------------------------------------------------------------------------
    // FieldSolver
    // ---------------------------------------------------------------------------

    FieldSolver::FieldSolver(std::shared_ptr<const mesh::Mesh<double>> mesh,
        std::shared_ptr<const mesh::MeshTags<int>> facet_tags,
        std::shared_ptr<const mesh::MeshTags<int>> cell_tags, int order,
        int value_dim)
        : mesh_(std::move(mesh))
        , facet_tags_(std::move(facet_tags))
        , cell_tags_(std::move(cell_tags))
        , order_(order)
        , value_dim_(value_dim)
    {
        V_ = make_space(mesh_, order_, value_dim_);
        u_ = std::make_shared<fem::Function<double>>(V_);
        u_->x()->set(0.0);

        // Cell-based sparsity (all supported operators are cell integrals
        // plus boundary facet integrals on shared dofs): the cell graph plus
        // the diagonal covers every entry.
        const auto cells = this->cells();
        pattern_ = std::make_shared<la::SparsityPattern>(
            V_->dofmap()->index_map, V_->dofmap()->index_map_bs());
        fem::sparsitybuild::cells(*pattern_, std::pair {cells, cells},
            {*V_->dofmap(), *V_->dofmap()});
        std::vector<std::int32_t> diag(V_->dofmap()->index_map->size_local());
        std::iota(diag.begin(), diag.end(), 0);
        pattern_->insert_diagonal(std::span(diag));
        pattern_->finalize();
    }

    std::vector<std::int32_t> FieldSolver::cells() const
    {
        std::vector<std::int32_t> cells(V_->dofmap()->map().extent(0));
        std::iota(cells.begin(), cells.end(), 0);
        return cells;
    }

    std::vector<std::int32_t> FieldSolver::boundary_dofs(
        const std::set<int>& ids) const
    {
        std::vector<std::int32_t> facets;
        if (facet_tags_)
            facets = tagged_facets(*facet_tags_, ids);
        if (facets.empty())
            return {};
        const int tdim = mesh_->topology()->dim();
        auto topo = mesh_->topology_mutable();
        topo->create_entities(tdim - 1);
        topo->create_connectivity(tdim - 1, tdim);
        topo->create_connectivity(tdim, tdim - 1);
        return fem::DirichletBC<double>::locate_dofs_topological(
            *mesh_->topology(), *V_->dofmap(), tdim - 1, facets);
    }

    std::vector<std::int32_t> FieldSolver::boundary_facets(
        const std::set<int>& ids) const
    {
        const int tdim = mesh_->topology()->dim();
        auto topo = mesh_->topology_mutable();
        topo->create_entities(tdim - 1);
        topo->create_connectivity(tdim, tdim - 1);
        topo->create_connectivity(tdim - 1, tdim);
        auto c_to_f = topo->connectivity(tdim, tdim - 1);
        auto e_to_c = topo->connectivity(tdim - 1, tdim);

        // Flattened (cell, local facet) pairs; the cell-local facet index
        // comes from the (tdim, tdim-1) connectivity.
        std::vector<std::int32_t> entities;
        if (!facet_tags_)
            return entities;
        for (std::int32_t f : tagged_facets(*facet_tags_, ids)) {
            if (e_to_c->num_links(f) != 1) // interior facet
                continue;
            const std::int32_t c = e_to_c->links(f)[0];
            for (int k = 0; k < c_to_f->num_links(c); ++k)
                if (c_to_f->links(c)[k] == f) {
                    entities.push_back(c);
                    entities.push_back(k);
                    break;
                }
        }
        return entities;
    }

    std::vector<fem::DirichletBC<double>> FieldSolver::make_bcs(
        const std::map<int, ScalarExpression>& values, double t) const
    {
        std::vector<fem::DirichletBC<double>> bcs;
        for (const auto& [boundary_id, value] : values) {
            auto dofs = boundary_dofs({boundary_id});
            if (not dofs.empty())
                bcs.emplace_back(value.eval(0, 0, 0, t), dofs, V_);
        }
        return bcs;
    }

    FieldSolver::DirichletRows FieldSolver::marked_rows(
        const std::vector<fem::DirichletBC<double>>& bcs) const
    {
        const std::size_t n = static_cast<std::size_t>(V_->dofmap()->index_map_bs())
            * static_cast<std::size_t>(V_->dofmap()->index_map->size_local());
        DirichletRows rows(n, 0);
        for (const auto& bc : bcs)
            bc.mark_dofs(rows);
        return rows;
    }

    std::shared_ptr<const fem::PrecomputeData<double>> FieldSolver::cell_precompute(
        const Coefficients& coeffs) const
    {
        return std::make_shared<fem::PrecomputeData<double>>(
            mesh_->topology()->cell_type(), *V_->element(), *V_->element(),
            coefficient_elements(coeffs), mesh_->geometry().cmaps().front(), 2);
    }

    std::shared_ptr<const fem::FacetPrecomputeData<double>>
    FieldSolver::facet_precompute(const Coefficients& coeffs) const
    {
        return std::make_shared<fem::FacetPrecomputeData<double>>(
            mesh_->topology()->cell_type(), *V_->element(), *V_->element(),
            coefficient_elements(coeffs), mesh_->geometry().cmaps().front(), 2);
    }

    void FieldSolver::assemble_into(la::MatrixCSR<double>& A,
        const fem::Form<double>& a, const DirichletRows& bc_rows) const
    {
        if (V_->dofmap()->index_map_bs() == 1)
            fem::assemble_matrix(A.mat_add_values(), a, bc_rows, {});
        else
            fem::assemble_matrix(A.mat_add_values<3, 3>(), a, bc_rows, {});
    }

    fem::Form<double> FieldSolver::add_operator(la::MatrixCSR<double>& A,
        const DirichletRows& bc_rows, const Coefficients& coeffs,
        fem::cell_kernel_weak_fn_t<double> w) const
    {
        const auto cells = this->cells();
        std::vector<std::shared_ptr<const fem::FunctionSpace<double>>> spaces {V_, V_};
        fem::Form<double> a(spaces,
            single_integral(fem::IntegralType::cell, cells,
                keeping_cell_kernel(cell_precompute(coeffs), w), coeffs.size()),
            mesh_, coeffs, {});
        assemble_into(A, a, bc_rows);
        return a;
    }

    fem::Form<double> FieldSolver::add_operator(la::MatrixCSR<double>& A,
        const DirichletRows& bc_rows, const Coefficients& coeffs,
        fem::facet_kernel_weak_fn_t<double> w, int boundary) const
    {
        const auto entities = boundary_facets({boundary});
        std::vector<std::shared_ptr<const fem::FunctionSpace<double>>> spaces {V_, V_};
        fem::Form<double> a(spaces,
            single_integral(fem::IntegralType::exterior_facet, entities,
                keeping_facet_kernel(facet_precompute(coeffs), w), coeffs.size()),
            mesh_, coeffs, {});
        assemble_into(A, a, bc_rows);
        return a;
    }

    void FieldSolver::add_load(la::Vector<double>& b, const Coefficients& coeffs,
        fem::cell_kernel_weak_fn_t<double> w, Constants constants) const
    {
        const auto cells = this->cells();
        std::vector<std::shared_ptr<const fem::FunctionSpace<double>>> spaces {V_};
        fem::Form<double> L(spaces,
            single_integral(fem::IntegralType::cell, cells,
                keeping_cell_kernel(cell_precompute(coeffs), w), coeffs.size()),
            mesh_, coeffs, std::move(constants));
        fem::assemble_vector(b, L);
    }

    void FieldSolver::add_load(la::Vector<double>& b, const Coefficients& coeffs,
        fem::facet_kernel_weak_fn_t<double> w, int boundary) const
    {
        const auto entities = boundary_facets({boundary});
        if (entities.empty())
            return;
        std::vector<std::shared_ptr<const fem::FunctionSpace<double>>> spaces {V_};
        fem::Form<double> L(spaces,
            single_integral(fem::IntegralType::exterior_facet, entities,
                keeping_facet_kernel(facet_precompute(coeffs), w), coeffs.size()),
            mesh_, coeffs, {});
        fem::assemble_vector(b, L);
    }

    // ---------------------------------------------------------------------------
    // ElectrostaticsSolver
    // ---------------------------------------------------------------------------

    ElectrostaticsSolver::ElectrostaticsSolver(
        std::shared_ptr<const mesh::Mesh<double>> mesh,
        std::shared_ptr<const mesh::MeshTags<int>> facet_tags,
        std::shared_ptr<const mesh::MeshTags<int>> cell_tags, int order)
        : FieldSolver(std::move(mesh), std::move(facet_tags), std::move(cell_tags),
              order, 1)
    {
    }

    void ElectrostaticsSolver::refresh(double t)
    {
        t_ = t;
        if (sigma_)
            sigma_->update(t);
    }

    void ElectrostaticsSolver::assemble_steady(la::MatrixCSR<double>& A,
        la::Vector<double>& b) const
    {
        auto bcs = make_bcs(voltages_, t_);
        auto rows = marked_rows(bcs);
        auto a = add_operator(A, rows, {sigma_->function()},
            kernels::diffusion_scalar);
        b.set(0.0);
        impose_dirichlet(A, b, a, bcs, rows);
    }

    void ElectrostaticsSolver::constrain_solution(double t)
    {
        impose_values(std::span(u_->x()->array()), make_bcs(voltages_, t));
    }

    // ---------------------------------------------------------------------------
    // HeatTransferSolver
    // ---------------------------------------------------------------------------

    HeatTransferSolver::HeatTransferSolver(
        std::shared_ptr<const mesh::Mesh<double>> mesh,
        std::shared_ptr<const mesh::MeshTags<int>> facet_tags,
        std::shared_ptr<const mesh::MeshTags<int>> cell_tags, int order)
        : FieldSolver(std::move(mesh), std::move(facet_tags), std::move(cell_tags),
              order, 1)
    {
    }

    void HeatTransferSolver::set_joule_source(
        std::shared_ptr<const fem::Function<double>> V,
        std::shared_ptr<CellProperty> sigma)
    {
        joule_V_ = std::move(V);
        joule_sigma_ = std::move(sigma);
    }

    void HeatTransferSolver::refresh(double t)
    {
        t_ = t;
        if (k_)
            k_->update(t);
        if (rho_cp_)
            rho_cp_->update(t);
        if (Q_)
            Q_->update(t);
        if (joule_sigma_)
            joule_sigma_->update(t);
        for (auto& cv : convections_) {
            cv.h->update(t);
            cv.t_inf->update(t);
        }
    }

    bool HeatTransferSolver::nonlinear() const
    {
        return solution_dependent(k_) or solution_dependent(rho_cp_)
            or solution_dependent(Q_) or solution_dependent(joule_sigma_);
    }

    void HeatTransferSolver::constrain_solution(double t)
    {
        impose_values(std::span(u_->x()->array()), make_bcs(temps_, t));
    }

    void HeatTransferSolver::apply_initial_condition()
    {
        const auto coords = V_->tabulate_dof_coordinates(false);
        auto& arr = u_->x()->array();
        for (std::int32_t d = 0; d < V_->dofmap()->index_map->size_local(); ++d)
            arr[static_cast<std::size_t>(d)] = initial_.eval(
                coords[3 * d], coords[3 * d + 1], coords[3 * d + 2], 0.0);
    }

    void HeatTransferSolver::assemble_sources(la::Vector<double>& f) const
    {
        if (Q_)
            add_load(f, {Q_->function()}, kernels::load_scalar);

        // Joule heating: ∫ sigma |grad V|² phi.
        if (joule_V_ and joule_sigma_)
            add_load(f, {joule_sigma_->function(), joule_V_},
                kernels::joule_heat_load);

        // Convection load h Tinf.
        for (const auto& cv : convections_)
            add_load(f, {cv.h->function(), cv.t_inf->function()},
                kernels::convection_load, cv.boundary_id);
    }

    void HeatTransferSolver::assemble_step(la::MatrixCSR<double>& A,
        la::Vector<double>& b, const TimeLevel& level) const
    {
        const TimeWeights& w = level.weights;
        const double t = level.time;
        const auto& history = level.history;
        const la::Vector<double>* f_old = level.source_old;
        la::Vector<double>& f_new = *level.source_new;
        const auto& dofmap = *V_->dofmap();
        const int bs = dofmap.index_map_bs();
        const bool has_mass = w.a[0] != 0.0;

        auto bcs = make_bcs(temps_, t);
        auto rows = marked_rows(bcs);

        // Operators, assembled with the Dirichlet rows zeroed and their
        // columns kept (the boundary values move to the right-hand side in
        // impose_dirichlet).
        la::MatrixCSR<double> M(*pattern_);
        la::MatrixCSR<double> K(*pattern_);
        if (has_mass)
            add_operator(M, rows, {rho_cp_->function()}, kernels::mass_scalar);
        auto a = add_operator(K, rows, {k_->function()},
            kernels::diffusion_scalar);

        // Convection Robin mass.
        for (const auto& cv : convections_)
            add_operator(K, rows, {cv.h->function()}, kernels::convection_mass,
                cv.boundary_id);

        // LHS operator: a0 M + b0 K.
        auto& av = A.values();
        if (has_mass)
            for (std::size_t i = 0; i < av.size(); ++i)
                av[i] = w.a[0] * M.values()[i] + w.b[0] * K.values()[i];
        else
            for (std::size_t i = 0; i < av.size(); ++i)
                av[i] = w.b[0] * K.values()[i];

        // Source load of this level.
        f_new.set(0.0);
        assemble_sources(f_new);
        b.set(0.0);
        for (std::size_t i = 0; i < b.array().size(); ++i)
            b.array()[i] = w.c_new * f_new.array()[i]
                + (w.c_old != 0.0 and f_old ? w.c_old * f_old->array()[i] : 0.0);

        // History: b -= (a_k M + b_k K) u^{n+1-k}.
        la::Vector<double> y(dofmap.index_map, bs);
        for (std::size_t k = 1; k < w.a.size(); ++k) {
            if (k > history.size() or history[k - 1] == nullptr)
                break;
            const la::Vector<double>& u_k = *history[k - 1];
            if (w.a[k] != 0.0 and has_mass) {
                y.set(0.0);
                M.mult(u_k, y);
                for (std::size_t i = 0; i < b.array().size(); ++i)
                    b.array()[i] -= w.a[k] * y.array()[i];
            }
            if (w.b[k] != 0.0) {
                y.set(0.0);
                K.mult(u_k, y);
                for (std::size_t i = 0; i < b.array().size(); ++i)
                    b.array()[i] -= w.b[k] * y.array()[i];
            }
        }

        impose_dirichlet(A, b, a, bcs, rows);
    }

    void HeatTransferSolver::assemble_steady(la::MatrixCSR<double>& A,
        la::Vector<double>& b) const
    {
        // The steady problem is the degenerate one-level scheme: K u = f.
        TimeWeights weights;
        weights.a = {0.0};
        weights.b = {1.0};
        la::Vector<double> f_new(V_->dofmap()->index_map,
            V_->dofmap()->index_map_bs());
        TimeLevel level;
        level.weights = weights;
        level.time = t_;
        level.source_new = &f_new;
        assemble_step(A, b, level);
    }

    // ---------------------------------------------------------------------------
    // SolidMechanicsSolver
    // ---------------------------------------------------------------------------

    SolidMechanicsSolver::SolidMechanicsSolver(
        std::shared_ptr<const mesh::Mesh<double>> mesh,
        std::shared_ptr<const mesh::MeshTags<int>> facet_tags,
        std::shared_ptr<const mesh::MeshTags<int>> cell_tags, int order)
        : FieldSolver(std::move(mesh), std::move(facet_tags), std::move(cell_tags),
              order, 3)
    {
    }

    void SolidMechanicsSolver::set_thermal_expansion(
        std::shared_ptr<const fem::Function<double>> T,
        std::shared_ptr<CellProperty> alpha, double t_ref)
    {
        thermal_ = Thermal {std::move(T), std::move(alpha), t_ref};
    }

    void SolidMechanicsSolver::refresh(double t)
    {
        t_ = t;
        if (E_)
            E_->update(t);
        if (nu_)
            nu_->update(t);
        if (thermal_)
            thermal_->alpha->update(t);
    }

    bool SolidMechanicsSolver::nonlinear() const
    {
        return solution_dependent(E_) or solution_dependent(nu_)
            or (thermal_ and solution_dependent(thermal_->alpha));
    }

    void SolidMechanicsSolver::assemble_steady(la::MatrixCSR<double>& A,
        la::Vector<double>& b) const
    {
        auto bcs = fixed_bcs();
        auto rows = marked_rows(bcs);
        auto a = add_operator(A, rows, {E_->function(), nu_->function()},
            kernels::elasticity);

        // RHS: thermal expansion load ∫ Bᵀ sigma_th.
        b.set(0.0);
        if (thermal_) {
            auto t_ref = std::make_shared<fem::Constant<double>>(thermal_->t_ref);
            add_load(b,
                {thermal_->T, thermal_->alpha->function(), E_->function(),
                    nu_->function()},
                kernels::thermal_expansion_load, {std::move(t_ref)});
        }

        impose_dirichlet(A, b, a, bcs, rows);
    }

    std::vector<fem::DirichletBC<double>> SolidMechanicsSolver::fixed_bcs() const
    {
        // A fixed boundary fixes every component of its dofs.
        std::vector<std::int32_t> dofs = boundary_dofs(fixed_);
        std::ranges::sort(dofs);
        dofs.erase(std::unique(dofs.begin(), dofs.end()), dofs.end());
        std::vector<std::int32_t> physical;
        physical.reserve(dofs.size() * 3);
        for (std::int32_t d : dofs)
            for (int k = 0; k < 3; ++k)
                physical.push_back(3 * d + k);
        std::vector<fem::DirichletBC<double>> bcs;
        if (not physical.empty())
            bcs.emplace_back(0.0, physical, V_);
        return bcs;
    }

    void SolidMechanicsSolver::constrain_solution(double t)
    {
        (void)t; // a fixed boundary prescribes a constant zero displacement
        impose_values(std::span(u_->x()->array()), fixed_bcs());
    }

} // namespace hellofem::app
