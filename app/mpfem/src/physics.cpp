// hellofem::app — physics field solvers (electrostatics / heat / solid)
// SPDX-License-Identifier: MIT

#include "physics.h"

#include "basis/element-families.h"
#include "fem/CoordinateElement.h"
#include "fem/DirichletBC.h"
#include "fem/DofMap.h"
#include "fem/Form.h"
#include "fem/assembler.h"
#include "fem/dofmapbuilder.h"
#include "fem/facet_precompute.h"
#include "fem/precompute.h"
#include "fem/sparsitybuild.h"
#include "kernels.h"
#include "la/KrylovSolver.h"
#include "la/SparsityPattern.h"
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
    using C = basis::cell::type;
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
        auto coord = mesh->geometry().cmaps().front();
        auto layout = coord.create_dof_layout();
        auto [imap, bs, dofmaps] = fem::build_dofmap_data(*mesh->topology(),
            {layout}, nullptr);
        auto dmap = std::make_shared<fem::DofMap>(layout,
            std::make_shared<common::IndexMap>(std::move(imap)), bs,
            std::move(dofmaps.front()), bs);
        return std::make_shared<fem::FunctionSpace<double>>(mesh, fe, dmap);
    }

    std::vector<std::int32_t> all_cells(const fem::DofMap& dmap)
    {
        std::vector<std::int32_t> cells(dmap.map().extent(0));
        std::iota(cells.begin(), cells.end(), 0);
        return cells;
    }

    /// Boundary facets carrying the given 1-based COMSOL boundary id.
    std::vector<std::int32_t> facets_by_id(
        const std::shared_ptr<const mesh::MeshTags<int>>& tags, int id)
    {
        std::vector<std::int32_t> out;
        if (!tags)
            return out;
        for (std::size_t i = 0; i < tags->indices().size(); ++i)
            if (tags->values()[i] == id)
                out.push_back(tags->indices()[i]);
        return out;
    }

    /// Locate the dofs on the given boundary facets (dimension tdim-1).
    std::vector<std::int32_t> boundary_dofs(const mesh::Mesh<double>& mesh,
        const fem::DofMap& dmap, std::span<const std::int32_t> facets)
    {
        if (facets.empty())
            return {};
        return fem::DirichletBC<double>::locate_dofs_topological(
            *mesh.topology(), dmap, mesh.topology()->dim() - 1, facets);
    }

    /// Flattened `(cell, local_facet)` pairs for the boundary facets with
    /// the given ids. The cell-local facet index is read from the
    /// `(tdim, tdim-1)` connectivity.
    std::vector<std::int32_t> boundary_facet_entities(
        const mesh::Mesh<double>& mesh,
        const std::shared_ptr<const mesh::MeshTags<int>>& tags,
        const std::set<int>& ids)
    {
        const int tdim = mesh.topology()->dim();
        auto topo = mesh.topology_mutable();
        topo->create_entities(tdim - 1);
        topo->create_connectivity(tdim, tdim - 1);
        topo->create_connectivity(tdim - 1, tdim);
        auto c_to_f = topo->connectivity(tdim, tdim - 1);
        auto e_to_c = topo->connectivity(tdim - 1, tdim);

        std::vector<std::int32_t> entities;
        if (!tags)
            return entities;
        for (std::size_t i = 0; i < tags->indices().size(); ++i) {
            if (not ids.contains(tags->values()[i]))
                continue;
            const std::int32_t f = tags->indices()[i];
            // The facet belongs to exactly one cell (exterior facet).
            if (e_to_c->num_links(f) != 1)
                continue;
            const std::int32_t c = e_to_c->links(f)[0];
            int lf = -1;
            for (int k = 0; k < c_to_f->num_links(c); ++k)
                if (c_to_f->links(c)[k] == f) {
                    lf = k;
                    break;
                }
            if (lf >= 0) {
                entities.push_back(c);
                entities.push_back(lf);
            }
        }
        return entities;
    }

} // namespace

// ---------------------------------------------------------------------------
// CellProperty
// ---------------------------------------------------------------------------

CellProperty::CellProperty(std::shared_ptr<const mesh::Mesh<double>> mesh,
    std::shared_ptr<const mesh::MeshTags<int>> cell_tags)
    : mesh_(std::move(mesh))
    , cell_tags_(std::move(cell_tags))
{
    // DG0 space: one scalar dof per cell.
    auto cell_type = mesh_->topology()->cell_type();
    auto ctype = mesh::cell_type_to_basix_type(cell_type);
    auto fe = std::make_shared<fem::FiniteElement<double>>(
        basis::create_element<double>(B::P, ctype, 0, LV::equispaced, DV::unset,
            true)); // discontinuous P0
    // P0 has one dof per cell; use the element's own dof layout (not the
    // coordinate element's P1 layout).
    auto layout = fe->create_dof_layout();
    auto [imap, bs, dofmaps] = fem::build_dofmap_data(*mesh_->topology(),
        {layout}, nullptr);
    auto dmap = std::make_shared<fem::DofMap>(layout,
        std::make_shared<common::IndexMap>(std::move(imap)), bs,
        std::move(dofmaps.front()), bs);
    auto V = std::make_shared<fem::FunctionSpace<double>>(mesh_, fe, dmap);
    f_ = std::make_shared<fem::Function<double>>(V);
    f_->x()->set(0.0);
}

void CellProperty::set_domain(int dom, double value)
{
    auto& arr = f_->x()->array();
    if (cell_tags_) {
        for (std::size_t i = 0; i < cell_tags_->indices().size(); ++i)
            if (cell_tags_->values()[i] == dom)
                arr[static_cast<std::size_t>(cell_tags_->indices()[i])] = value;
    }
    else {
        std::fill(arr.begin(), arr.end(), value);
    }
}

void CellProperty::fill_from(std::function<double(double, double, double, double)> eval,
    const std::set<int>& domains, double t)
{
    auto c_to_v = mesh_->topology()->connectivity(mesh_->topology()->dim(), 0);
    const auto [vc, vshape] = mesh::compute_vertex_coords(*mesh_);
    const std::size_t nv = vshape[1];
    auto& arr = f_->x()->array();
    for (std::int32_t c = 0; c < static_cast<std::int32_t>(c_to_v->num_nodes()); ++c) {
        if (cell_tags_ and not domains.empty()
            and not domains.contains(cell_tags_->values()[c]))
            continue;
        auto verts = c_to_v->links(c);
        std::array<double, 3> xc {0, 0, 0};
        for (auto v : verts)
            for (int d = 0; d < 3; ++d)
                xc[d] += vc[d * nv + v];
        for (int d = 0; d < 3; ++d)
            xc[d] /= static_cast<double>(verts.size());
        arr[static_cast<std::size_t>(c)] = eval(xc[0], xc[1], xc[2], t);
    }
}

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

void ElectrostaticsSolver::solve_steady()
{
    auto cells = all_cells(*V_->dofmap());
    auto coord = mesh_->geometry().cmaps().front();
    auto pre = std::make_shared<fem::PrecomputeData<double>>(
        mesh_->topology()->cell_type(), *V_->element(), *V_->element(),
        std::vector<const fem::FiniteElement<double>*> {sigma_->function()->function_space()->element().get()},
        coord, 2);
    auto kernel = fem::make_cell_kernel(*pre, kernels::diffusion_scalar);
    std::vector<std::shared_ptr<const fem::FunctionSpace<double>>> Vlist {V_, V_};
    std::map<std::pair<fem::IntegralType, int>, std::vector<fem::Form<double>::integral_data>> integrals;
    integrals[{fem::IntegralType::cell, 0}] = {{kernel, cells, {0}}};
    std::vector<std::shared_ptr<const fem::Function<double>>> coeffs {
        std::make_shared<fem::Function<double>>(*sigma_->function())};
    fem::Form<double> a(Vlist, std::move(integrals), mesh_, coeffs, {});

    // Sparsity pattern + matrix.
    la::SparsityPattern pattern(V_->dofmap()->index_map, 1);
    fem::sparsitybuild::cells(pattern, std::pair {cells, cells}, {*V_->dofmap(), *V_->dofmap()});
    std::vector<std::int32_t> diag(V_->dofmap()->index_map->size_local());
    std::iota(diag.begin(), diag.end(), 0);
    pattern.insert_diagonal(std::span(diag));
    pattern.finalize();
    la::MatrixCSR<double> A(pattern);

    // RHS: no volume source -> zero.
    la::Vector<double> b(V_->dofmap()->index_map, V_->dofmap()->index_map_bs());
    b.set(0.0);

    // Dirichlet BCs: one BC per voltage boundary group (each with a value).
    std::vector<fem::DirichletBC<double>> bcs;
    bcs.reserve(voltages_.size());
    for (const auto& [bid, value] : voltages_) {
        auto facets = facets_by_id(facet_tags_, bid);
        auto dofs = boundary_dofs(*mesh_, *V_->dofmap(), facets);
        if (!dofs.empty())
            bcs.emplace_back(value, dofs, V_);
    }

    if (!bcs.empty()) {
        std::vector<std::reference_wrapper<const fem::DirichletBC<double>>> bref;
        for (auto& b : bcs)
            bref.emplace_back(b);
        fem::assemble_matrix(A.mat_add_values(), a, bref);
        fem::set_diagonal(A.mat_set_values(), a, bref, 1.0);
        fem::apply_lifting(b, a, bref, std::optional<std::span<const double>> {}, 1.0);
        fem::set_bc(std::span(b.array()), bref, std::optional<std::span<const double>> {}, 1.0);
    }
    else {
        fem::assemble_matrix(A.mat_add_values(), a, {});
    }

    la::KrylovSolver<double> solver;
    solver.set_operator(A);
    solver.set_solver_type("cg");
    solver.set_tolerances(1e-12, 1e-14, 2000);
    solver.solve(*u_->x(), b);
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

void HeatTransferSolver::add_convection(int boundary_id, double h, double Tinf)
{
    convections_.push_back({boundary_id, h, Tinf});
}

void HeatTransferSolver::solve_steady()
{
    auto cells = all_cells(*V_->dofmap());
    auto coord = mesh_->geometry().cmaps().front();

    // Dirichlet BCs: one BC per temperature boundary group.
    std::vector<fem::DirichletBC<double>> bcs;
    for (const auto& [bid, value] : temps_) {
        auto facets = facets_by_id(facet_tags_, bid);
        auto dofs = boundary_dofs(*mesh_, *V_->dofmap(), facets);
        if (!dofs.empty())
            bcs.emplace_back(value, dofs, V_);
    }
    std::vector<std::reference_wrapper<const fem::DirichletBC<double>>> bref;
    for (auto& bc : bcs)
        bref.emplace_back(bc);

    // K = diffusion(k) + convection Robin mass.
    auto pre = std::make_shared<fem::PrecomputeData<double>>(
        mesh_->topology()->cell_type(), *V_->element(), *V_->element(),
        std::vector<const fem::FiniteElement<double>*> {k_->function()->function_space()->element().get()},
        coord, 2);
    auto kernel = fem::make_cell_kernel(*pre, kernels::diffusion_scalar);
    std::vector<std::shared_ptr<const fem::FunctionSpace<double>>> Vlist {V_, V_};
    std::map<std::pair<fem::IntegralType, int>, std::vector<fem::Form<double>::integral_data>> integrals;
    integrals[{fem::IntegralType::cell, 0}] = {{kernel, cells, {0}}};
    std::vector<std::shared_ptr<const fem::Function<double>>> coeffs {
        std::make_shared<fem::Function<double>>(*k_->function())};
    fem::Form<double> a(Vlist, integrals, mesh_, coeffs, {});

    // Sparsity.
    la::SparsityPattern pattern(V_->dofmap()->index_map, 1);
    fem::sparsitybuild::cells(pattern, std::pair {cells, cells}, {*V_->dofmap(), *V_->dofmap()});
    std::vector<std::int32_t> diag(V_->dofmap()->index_map->size_local());
    std::iota(diag.begin(), diag.end(), 0);
    pattern.insert_diagonal(std::span(diag));
    pattern.finalize();
    la::MatrixCSR<double> A(pattern);

    // Assemble diffusion (with BC zeroing) and the convection facet mass.
    fem::assemble_matrix(A.mat_add_values(), a, bref);

    // RHS: heat source + convection load.
    la::Vector<double> b(V_->dofmap()->index_map, V_->dofmap()->index_map_bs());
    b.set(0.0);

    // Convection facet contributions to K and F.
    mesh_->topology_mutable()->create_entities(mesh_->topology()->dim() - 1);
    for (const auto& cv : convections_) {
        std::set<int> ids {cv.id};
        auto entities = boundary_facet_entities(*mesh_, facet_tags_, ids);
        if (entities.empty())
            continue;

        // h as a DG0 cell property (constant on all cells).
        auto h_dg0 = std::make_shared<CellProperty>(mesh_, cell_tags_);
        h_dg0->set_domain(1, cv.h);

        // K += h * phi phi' on this boundary (with BC zeroing).
        auto pre_conv = std::make_shared<fem::FacetPrecomputeData<double>>(
            mesh_->topology()->cell_type(), *V_->element(), *V_->element(),
            std::vector<const fem::FiniteElement<double>*> {h_dg0->function()->function_space()->element().get()},
            coord, 2);
        auto kconv = fem::make_facet_kernel(*pre_conv, kernels::convection_mass);
        std::map<std::pair<fem::IntegralType, int>, std::vector<fem::Form<double>::integral_data>> ints_conv;
        ints_conv[{fem::IntegralType::exterior_facet, 0}] = {
            {kconv, entities, {0}}};
        std::vector<std::shared_ptr<const fem::Function<double>>> hcoeffs {
            std::make_shared<fem::Function<double>>(*h_dg0->function())};
        fem::Form<double> a_conv(Vlist, std::move(ints_conv), mesh_, hcoeffs, {});
        fem::assemble_matrix(A.mat_add_values(), a_conv, bref);

        // b += h*Tinf * phi on this boundary.
        auto hT_dg0 = std::make_shared<CellProperty>(mesh_, cell_tags_);
        hT_dg0->set_domain(1, cv.h * cv.Tinf);
        auto pre_cl = std::make_shared<fem::FacetPrecomputeData<double>>(
            mesh_->topology()->cell_type(), *V_->element(), *V_->element(),
            std::vector<const fem::FiniteElement<double>*> {hT_dg0->function()->function_space()->element().get()},
            coord, 2);
        auto kcl = fem::make_facet_kernel(*pre_cl, kernels::convection_load);
        std::map<std::pair<fem::IntegralType, int>, std::vector<fem::Form<double>::integral_data>> ints_cl;
        ints_cl[{fem::IntegralType::exterior_facet, 0}] = {{kcl, entities, {0}}};
        std::vector<std::shared_ptr<const fem::FunctionSpace<double>>> Vlist1 {V_};
        std::vector<std::shared_ptr<const fem::Function<double>>> hTcoeffs {
            std::make_shared<fem::Function<double>>(*hT_dg0->function())};
        fem::Form<double> L_conv(Vlist1, std::move(ints_cl), mesh_, hTcoeffs, {});
        fem::assemble_vector(b, L_conv);
    }

    // Heat source.
    if (Q_) {
        auto preQ = std::make_shared<fem::PrecomputeData<double>>(
            mesh_->topology()->cell_type(), *V_->element(), *V_->element(),
            std::vector<const fem::FiniteElement<double>*> {Q_->function()->function_space()->element().get()},
            coord, 2);
        auto kload = fem::make_cell_kernel(*preQ, kernels::load_scalar);
        std::vector<std::shared_ptr<const fem::FunctionSpace<double>>> Vlist1 {V_};
        std::map<std::pair<fem::IntegralType, int>, std::vector<fem::Form<double>::integral_data>> ints;
        ints[{fem::IntegralType::cell, 0}] = {{kload, cells, {0}}};
        std::vector<std::shared_ptr<const fem::Function<double>>> qcoeffs {
            std::make_shared<fem::Function<double>>(*Q_->function())};
        fem::Form<double> L(Vlist1, std::move(ints), mesh_, qcoeffs, {});
        fem::assemble_vector(b, L);
    }

    // Apply Dirichlet: diagonal, lifting, RHS injection.
    if (!bcs.empty()) {
        fem::set_diagonal(A.mat_set_values(), a, bref, 1.0);
        fem::apply_lifting(b, a, bref, std::optional<std::span<const double>> {}, 1.0);
        fem::set_bc(std::span(b.array()), bref, std::optional<std::span<const double>> {}, 1.0);
    }

    la::KrylovSolver<double> solver;
    solver.set_operator(A);
    solver.set_solver_type("cg");
    solver.set_tolerances(1e-12, 1e-14, 2000);
    solver.solve(*u_->x(), b);
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

void SolidMechanicsSolver::solve_steady()
{
    auto cells = all_cells(*V_->dofmap());
    auto coord = mesh_->geometry().cmaps().front();

    // K = elasticity(E, nu).
    std::vector<const fem::FiniteElement<double>*> ce;
    if (E_)
        ce.push_back(E_->function()->function_space()->element().get());
    if (nu_)
        ce.push_back(nu_->function()->function_space()->element().get());
    auto pre = std::make_shared<fem::PrecomputeData<double>>(
        mesh_->topology()->cell_type(), *V_->element(), *V_->element(), ce, coord, 2);
    auto kernel = fem::make_cell_kernel(*pre, kernels::elasticity);
    std::vector<std::shared_ptr<const fem::FunctionSpace<double>>> Vlist {V_, V_};
    std::map<std::pair<fem::IntegralType, int>, std::vector<fem::Form<double>::integral_data>> integrals;
    integrals[{fem::IntegralType::cell, 0}] = {{kernel, cells, {0, 1}}};
    std::vector<std::shared_ptr<const fem::Function<double>>> coeffs {
        std::make_shared<fem::Function<double>>(*E_->function()),
        std::make_shared<fem::Function<double>>(*nu_->function())};
    fem::Form<double> a(Vlist, std::move(integrals), mesh_, coeffs, {});

    la::SparsityPattern pattern(V_->dofmap()->index_map, V_->dofmap()->index_map_bs());
    fem::sparsitybuild::cells(pattern, std::pair {cells, cells}, {*V_->dofmap(), *V_->dofmap()});
    std::vector<std::int32_t> diag(V_->dofmap()->index_map->size_local());
    std::iota(diag.begin(), diag.end(), 0);
    pattern.insert_diagonal(std::span(diag));
    pattern.finalize();
    la::MatrixCSR<double> A(pattern);
    fem::assemble_matrix(A.mat_add_values(), a, {});

    // RHS: thermal strain load.
    la::Vector<double> b(V_->dofmap()->index_map, V_->dofmap()->index_map_bs());
    b.set(0.0);
    if (sigma_th_) {
        auto preL = std::make_shared<fem::PrecomputeData<double>>(
            mesh_->topology()->cell_type(), *V_->element(), *V_->element(),
            std::vector<const fem::FiniteElement<double>*> {},
            coord, 2);
        // Thermal strain is a per-cell constant 6-vector; assemble directly.
        // (Handled by the coupling layer with a custom loop.)
    }

    // Dirichlet fixed BCs: zero displacement on the fixed boundaries.
    std::vector<std::int32_t> bc_dofs;
    for (int id : fixed_) {
        auto facets = facets_by_id(facet_tags_, id);
        auto dofs = boundary_dofs(*mesh_, *V_->dofmap(), facets);
        bc_dofs.insert(bc_dofs.end(), dofs.begin(), dofs.end());
    }
    std::ranges::sort(bc_dofs);
    bc_dofs.erase(std::unique(bc_dofs.begin(), bc_dofs.end()), bc_dofs.end());

    if (!bc_dofs.empty()) {
        fem::DirichletBC<double> bc(0.0, bc_dofs, V_);
        std::vector<std::reference_wrapper<const fem::DirichletBC<double>>> bref {bc};
        fem::assemble_matrix(A.mat_add_values(), a, bref);
        fem::set_diagonal(A.mat_set_values(), a, bref, 1.0);
        fem::apply_lifting(b, a, bref, std::optional<std::span<const double>> {}, 1.0);
        fem::set_bc(std::span(b.array()), bref, std::optional<std::span<const double>> {}, 1.0);
    }

    la::KrylovSolver<double> solver;
    solver.set_operator(A);
    solver.set_solver_type("cg");
    solver.set_tolerances(1e-10, 1e-12, 2000);
    solver.solve(*u_->x(), b);
}

} // namespace hellofem::app
