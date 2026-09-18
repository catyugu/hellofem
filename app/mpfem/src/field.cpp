// hellofem::app — field solver base implementation
// SPDX-License-Identifier: MIT

#include "field.h"

#include "solver.h"

#include "basis/element-families.h"
#include "fem/CoordinateElement.h"
#include "fem/assembler.h"
#include "fem/dofmapbuilder.h"
#include "fem/facet_precompute.h"
#include "fem/precompute.h"
#include "fem/sparsitybuild.h"
#include "fem/utils.h"
#include "mesh/Topology.h"
#include "mesh/cell_types.h"
#include "mesh/utils.h"

#include <algorithm>
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
            return fem::create_functionspace(mesh, std::move(fe));
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

        /// One integral over the given cells or facets, with the given kernel
        /// and the coefficient indices the kernel reads.
        using Integrals = std::map<std::pair<fem::IntegralType, int>,
            std::vector<fem::Form<double>::integral_data>>;

        /// Degree of the quadrature rule of a cell or of a facet integral:
        /// the degree of the app's heaviest integrand. On an affine cell that
        /// integrates it exactly; on a curved isoparametric one it integrates
        /// the polynomial part, no rule being exact there.
        ///
        /// Every material property is a DG0 cell value, so an integrand is
        /// the basis of degree `p`, at most one field of degree `p`, and a
        /// constant. The heaviest is the Joule load
        /// `sigma |grad V|^2 phi_i`, of degree `3p - 2`; then the mass
        /// `c phi_i phi_j` at `2p`, the stiffness
        /// `k grad phi_i . grad phi_j` at `2p - 2`, and a source `f phi_i`
        /// at `p`. The cell and the facet path share the rule, so the two
        /// cannot drift apart.
        constexpr int quadrature_degree(int field_order)
        {
            return std::max(2 * field_order, 3 * field_order - 2);
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

        /// A weak-form kernel together with the reference data it reads by
        /// reference, kept alive for as long as the returned kernel lives.
        template <class Precompute>
        fem::kernel_t<double> keeping_alive(std::shared_ptr<const Precompute> pre,
            fem::kernel_t<double> kernel)
        {
            return [pre = std::move(pre),
                       kernel = std::move(kernel)](double* Ae, const double* coeffs,
                       const double* constants, const double* cds, const int* perm,
                       const std::uint8_t* entity, void* ctx) mutable {
                kernel(Ae, coeffs, constants, cds, perm, entity, ctx);
            };
        }

        /// The kernel of the cell weak form `w`, on the reference data of
        /// `coeffs`.
        fem::kernel_t<double> keeping_cell_kernel(
            std::shared_ptr<const fem::PrecomputeData<double>> pre,
            fem::cell_kernel_weak_fn_t<double> w)
        {
            auto kernel = fem::make_cell_kernel(*pre, w);
            return keeping_alive(std::move(pre), std::move(kernel));
        }

        /// Facet counterpart of `keeping_cell_kernel`.
        fem::kernel_t<double> keeping_facet_kernel(
            std::shared_ptr<const fem::FacetPrecomputeData<double>> pre,
            fem::facet_kernel_weak_fn_t<double> w)
        {
            auto kernel = fem::make_facet_kernel(*pre, w);
            return keeping_alive(std::move(pre), std::move(kernel));
        }

        /// The facets of the mesh and their neighbouring cells, created on
        /// demand: every boundary query below needs both connectivities.
        void ensure_facet_topology(const mesh::Mesh<double>& m)
        {
            const int tdim = m.topology()->dim();
            auto topo = m.topology_mutable();
            topo->create_entities(tdim - 1);
            topo->create_connectivity(tdim - 1, tdim);
            topo->create_connectivity(tdim, tdim - 1);
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
    {
        V_ = make_space(mesh_, order, value_dim);
        u_ = std::make_shared<fem::Function<double>>(V_);
        u_->x()->set(0.0);
        order_ = order;

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
        ensure_facet_topology(*mesh_);
        return fem::DirichletBC<double>::locate_dofs_topological(*mesh_->topology(),
            *V_->dofmap(), mesh_->topology()->dim() - 1, facets);
    }

    std::vector<std::int32_t> FieldSolver::boundary_facets(
        const std::set<int>& ids) const
    {
        if (!facet_tags_)
            return {};
        ensure_facet_topology(*mesh_);
        const int tdim = mesh_->topology()->dim();
        auto c_to_f = mesh_->topology()->connectivity(tdim, tdim - 1);
        auto e_to_c = mesh_->topology()->connectivity(tdim - 1, tdim);

        // Flattened (cell, local facet) pairs; the cell-local facet index
        // comes from the (tdim, tdim-1) connectivity.
        std::vector<std::int32_t> entities;
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
            coefficient_elements(coeffs), mesh_->geometry().cmaps().front(),
            quadrature_degree(order_));
    }

    std::shared_ptr<const fem::FacetPrecomputeData<double>>
    FieldSolver::facet_precompute(const Coefficients& coeffs) const
    {
        return std::make_shared<fem::FacetPrecomputeData<double>>(
            mesh_->topology()->cell_type(), *V_->element(), *V_->element(),
            coefficient_elements(coeffs), mesh_->geometry().cmaps().front(),
            quadrature_degree(order_));
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

    void FieldSolver::impose_dirichlet(la::MatrixCSR<double>& A,
        la::Vector<double>& b, const fem::Form<double>& a,
        const std::vector<fem::DirichletBC<double>>& bcs,
        const DirichletRows& marked) const
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

    void FieldSolver::impose_values(std::span<double> x,
        const std::vector<fem::DirichletBC<double>>& bcs) const
    {
        std::vector<std::reference_wrapper<const fem::DirichletBC<double>>> bref(
            bcs.begin(), bcs.end());
        fem::set_bc(x, bref, std::optional<std::span<const double>> {}, 1.0);
    }

    int FieldSolver::solve_steady(double t)
    {
        return solve_system(
            [&](la::MatrixCSR<double>& A, la::Vector<double>& b) {
                refresh(t);
                assemble_steady(A, b);
            },
            *u_->x(), *pattern_, linear_solver_, linear_);
    }

} // namespace hellofem::app
