// hellofem::fem — exterior and interior facet assembly tests
// SPDX-License-Identifier: MIT

#include "basis/element-families.h"
#include "basis/finite-element.h"
#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "common/IndexMap.h"
#include "fem/CoordinateElement.h"
#include "fem/DofMap.h"
#include "fem/FiniteElement.h"
#include "fem/Form.h"
#include "fem/FunctionSpace.h"
#include "fem/assembler.h"
#include "fem/dofmapbuilder.h"
#include "fem/facet_precompute.h"
#include "fem/precompute.h"
#include "fem/sparsitybuild.h"
#include "fem/utils.h"
#include "la/KrylovSolver.h"
#include "la/MatrixCSR.h"
#include "la/SparsityPattern.h"
#include "la/Vector.h"
#include "mesh/Mesh.h"
#include "mesh/Topology.h"
#include "mesh/cell_types.h"
#include "mesh/generation.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

using namespace hellofem;
using Catch::Approx;

namespace {

    using B = basis::element::family;
    using C = basis::cell::type;
    using LV = basis::element::lagrange_variant;
    using DV = basis::element::dpc_variant;

    /// Build a P1 FunctionSpace on an n-interval unit square.
    std::shared_ptr<fem::FunctionSpace<double>> p1_space(int n)
    {
        auto mesh = mesh::create_unit_square(n);
        auto fe = std::make_shared<fem::FiniteElement<double>>(
            basis::create_element<double>(
                B::P, C::triangle, 1, LV::equispaced, DV::unset, false));
        auto layout = fem::CoordinateElement<double>(
            mesh::CellType::triangle, 1, LV::equispaced)
                          .create_dof_layout();
        auto [imap, bs, dofmaps]
            = fem::build_dofmap_data(*mesh->topology(), {layout}, nullptr);
        auto dmap = std::make_shared<fem::DofMap>(layout,
            std::make_shared<common::IndexMap>(std::move(imap)), bs,
            std::move(dofmaps.front()), bs);
        return std::make_shared<fem::FunctionSpace<double>>(mesh, fe, dmap);
    }

    /// All cell indices of a dofmap as a vector.
    std::vector<std::int32_t> all_cells(const fem::DofMap& dmap)
    {
        std::vector<std::int32_t> cells(dmap.map().extent(0));
        for (std::size_t c = 0; c < cells.size(); ++c)
            cells[c] = static_cast<std::int32_t>(c);
        return cells;
    }

    /// Facet measure kernel: Ae[0] += w detJ (functional of 1 dS).
    void facet_measure_kernel(double* Ae, const fem::FacetKernelData<double>& data)
    {
        for (int p = 0; p < data.num_points; ++p)
            *Ae += data.detJ[p] * data.w[p];
    }

    /// Normal-x kernel: Ae[0] += w n_x.
    void facet_normal_x_kernel(double* Ae, const fem::FacetKernelData<double>& data)
    {
        for (int p = 0; p < data.num_points; ++p)
            *Ae += data.n[p * 3 + 0] * data.w[p];
    }

    /// Facet mass kernel (bilinear): Ae[i,j] += w detJ phi0_i phi1_j.
    void facet_mass_kernel(double* Ae, const fem::FacetKernelData<double>& data)
    {
        const int nd = data.num_dofs0;
        std::fill(Ae, Ae + nd * nd, 0);
        for (int p = 0; p < data.num_points; ++p)
            for (int i = 0; i < nd; ++i)
                for (int j = 0; j < nd; ++j)
                    Ae[i * nd + j] += data.w[p] * data.detJ[p]
                        * data.phi0[p * nd + i] * data.phi1[p * nd + j];
    }

    /// Stiffness kernel (cell).
    void stiffness_kernel(double* Ae, const fem::CellKernelData<double>& data)
    {
        const int nq = data.num_points;
        const int nd = data.num_dofs0;
        const int tdim = data.tdim;
        std::fill(Ae, Ae + nd * nd, 0);
        for (int q = 0; q < nq; ++q)
            for (int i = 0; i < nd; ++i)
                for (int j = 0; j < nd; ++j) {
                    double dot = 0;
                    for (int c = 0; c < tdim; ++c)
                        dot += data.dphi0[(q * nd + i) * tdim + c]
                            * data.dphi1[(q * nd + j) * tdim + c];
                    Ae[i * nd + j] += data.w[q] * data.detJ[q] * dot;
                }
    }

    /// Load kernel (cell): Ae[i] = sum_q w detJ phi_i.
    void load_kernel(double* Ae, const fem::CellKernelData<double>& data)
    {
        const int nq = data.num_points;
        const int nd = data.num_dofs0;
        std::fill(Ae, Ae + nd, 0);
        for (int q = 0; q < nq; ++q)
            for (int i = 0; i < nd; ++i)
                Ae[i] += data.w[q] * data.detJ[q] * data.phi0[q * nd + i];
    }

    /// Interior jump kernel: Ae = <jump u, jump v> over interior facets.
    void jump_kernel(double* Ae, const fem::FacetKernelData<double>& data)
    {
        const int nd = data.num_dofs0;
        const int nq = data.num_points;
        std::fill(Ae, Ae + 4 * nd * nd, 0);
        for (int p = 0; p < nq; ++p)
            for (int i = 0; i < 2 * nd; ++i)
                for (int j = 0; j < 2 * nd; ++j) {
                    const double si = (i < nd) ? 1.0 : -1.0;
                    const double sj = (j < nd) ? 1.0 : -1.0;
                    Ae[i * 2 * nd + j] += si * data.phi0[p * 2 * nd + i]
                        * sj * data.phi1[p * 2 * nd + j] * data.w[p]
                        * data.detJ[p];
                }
    }

    /// Build a one-integral form.
    template <typename Kernel>
    fem::Form<double> make_form(
        std::vector<std::shared_ptr<const fem::FunctionSpace<double>>> spaces,
        std::shared_ptr<const mesh::Mesh<double>> mesh,
        fem::IntegralType type, Kernel&& kernel,
        std::vector<std::int32_t> entities)
    {
        fem::Form<double>::integral_data data;
        data.kernel = std::forward<Kernel>(kernel);
        data.entities = std::move(entities);
        data.coeffs = {};
        std::map<std::pair<fem::IntegralType, int>,
            std::vector<fem::Form<double>::integral_data>>
            integrals;
        integrals[{type, 0}] = {data};
        return fem::Form<double>(std::move(spaces), std::move(integrals),
            std::move(mesh), {}, {});
    }

} // namespace

TEST_CASE("facet: exterior measure on the unit square is 4", "[fem]")
{
    auto V = p1_space(3);
    auto coord = V->mesh()->geometry().cmaps().front();
    fem::FacetPrecomputeData<double> pre(mesh::CellType::triangle,
        *V->element(), *V->element(), {}, coord, 2);

    auto entities = fem::exterior_facet_entities(*V->mesh()->topology());
    auto M = make_form({V}, V->mesh(), fem::IntegralType::exterior_facet,
        fem::make_facet_kernel(pre, facet_measure_kernel), entities);
    REQUIRE(fem::assemble_scalar(M) == Approx(4.0).margin(1e-10));
}

TEST_CASE("facet: normal boundary integral vanishes (divergence theorem)", "[fem]")
{
    auto V = p1_space(3);
    auto coord = V->mesh()->geometry().cmaps().front();
    fem::FacetPrecomputeData<double> pre(mesh::CellType::triangle,
        *V->element(), *V->element(), {}, coord, 2);

    auto entities = fem::exterior_facet_entities(*V->mesh()->topology());
    auto M = make_form({V}, V->mesh(), fem::IntegralType::exterior_facet,
        fem::make_facet_kernel(pre, facet_normal_x_kernel), entities);
    // For the unit square the x-normal boundary integral is zero.
    REQUIRE(fem::assemble_scalar(M) == Approx(0.0).margin(1e-10));
}

TEST_CASE("facet: exterior facet mass assembly is symmetric and sums to 4", "[fem]")
{
    auto V = p1_space(3);
    auto coord = V->mesh()->geometry().cmaps().front();
    fem::FacetPrecomputeData<double> pre(mesh::CellType::triangle,
        *V->element(), *V->element(), {}, coord, 2);

    auto entities = fem::exterior_facet_entities(*V->mesh()->topology());
    auto a = make_form({V, V}, V->mesh(), fem::IntegralType::exterior_facet,
        fem::make_facet_kernel(pre, facet_mass_kernel), entities);

    // Sparsity: boundary dofs are mesh vertices, already in the cell pattern.
    auto cells = all_cells(*V->dofmap());
    la::SparsityPattern pattern(V->dofmap()->index_map, 1);
    fem::sparsitybuild::cells(pattern, std::pair {cells, cells},
        {*V->dofmap(), *V->dofmap()});
    std::vector<std::int32_t> diag(V->dofmap()->index_map->size_local());
    for (std::int32_t d = 0; d < V->dofmap()->index_map->size_local(); ++d)
        diag[static_cast<std::size_t>(d)] = d;
    pattern.insert_diagonal(std::span(diag));
    pattern.finalize();

    la::MatrixCSR<double> A(pattern);
    fem::assemble_matrix(A.mat_add_values(), a, std::vector<std::reference_wrapper<const fem::DirichletBC<double>>>{});

    // Sum of all entries = <1, 1>_boundary = perimeter = 4.
    double total = 0;
    for (auto v : A.values())
        total += v;
    REQUIRE(total == Approx(4.0).margin(1e-10));

    // Symmetry.
    auto dense = A.to_dense();
    const std::size_t N = static_cast<std::size_t>(V->dofmap()->index_map->size_local());
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j)
            REQUIRE(dense[i * N + j] == Approx(dense[j * N + i]).margin(1e-10));
}

TEST_CASE("facet: Robin BC -Delta u = 1 with du/dn + u = 0", "[fem]")
{
    // Weak form: (grad u, grad v) + <u, v>_boundary = (1, v).
    // Taking v = 1 (a P1 basis function sum), the equation collapses to
    //   <u, 1>_boundary = (1, 1) = area = 1
    // (grad of a constant vanishes, so the stiffness term drops out).
    // Test that the assembled solution satisfies this exactly (a check of
    // the facet mass assembly consistency).
    const int n = 6;
    auto V = p1_space(n);
    auto mesh = V->mesh();
    auto coord = mesh->geometry().cmaps().front();
    auto cells = all_cells(*V->dofmap());

    fem::PrecomputeData<double> pre_stiff(mesh::CellType::triangle,
        *V->element(), *V->element(), {}, coord, 2);
    auto k_stiff = fem::make_cell_kernel(pre_stiff, stiffness_kernel);
    fem::PrecomputeData<double> pre_load(mesh::CellType::triangle,
        *V->element(), *V->element(), {}, coord, 2);
    auto k_load = fem::make_cell_kernel(pre_load, load_kernel);

    fem::FacetPrecomputeData<double> pre_facet(mesh::CellType::triangle,
        *V->element(), *V->element(), {}, coord, 2);
    auto k_facet = fem::make_facet_kernel(pre_facet, facet_mass_kernel);
    auto facet_entities = fem::exterior_facet_entities(*mesh->topology());

    fem::Form<double>::integral_data stiff, facet;
    stiff.kernel = k_stiff;
    stiff.entities = cells;
    stiff.coeffs = {};
    facet.kernel = k_facet;
    facet.entities = facet_entities;
    facet.coeffs = {};
    std::map<std::pair<fem::IntegralType, int>,
        std::vector<fem::Form<double>::integral_data>>
        integrals;
    integrals[{fem::IntegralType::cell, 0}] = {stiff};
    integrals[{fem::IntegralType::exterior_facet, 0}] = {facet};
    std::vector<std::shared_ptr<const fem::FunctionSpace<double>>> Vb {V, V};
    fem::Form<double> Aform(Vb, std::move(integrals), mesh, {}, {});

    la::SparsityPattern pattern(V->dofmap()->index_map, 1);
    fem::sparsitybuild::cells(pattern, std::pair {cells, cells},
        {*V->dofmap(), *V->dofmap()});
    std::vector<std::int32_t> diag(V->dofmap()->index_map->size_local());
    for (std::int32_t d = 0; d < V->dofmap()->index_map->size_local(); ++d)
        diag[static_cast<std::size_t>(d)] = d;
    pattern.insert_diagonal(std::span(diag));
    pattern.finalize();
    la::MatrixCSR<double> A(pattern);
    fem::assemble_matrix(A.mat_add_values(), Aform,
        std::vector<std::reference_wrapper<const fem::DirichletBC<double>>> {});

    auto L = make_form({V}, mesh, fem::IntegralType::cell, k_load, cells);
    la::Vector<double> b(V->dofmap()->index_map, V->dofmap()->index_map_bs());
    fem::assemble_vector(b, L);

    la::Vector<double> u(V->dofmap()->index_map, V->dofmap()->index_map_bs());
    la::KrylovSolver<double> solver;
    solver.set_operator(A);
    solver.set_solver_type("cg");
    solver.set_tolerances(1e-12, 1e-14, 1000);
    const int iters = solver.solve(u, b);
    REQUIRE(iters > 0);

    // <u, 1>_boundary = sum_i u_i * (facet mass row sum) must equal -1.
    // The facet mass matrix row sums give the integral of 1 over the
    // boundary dofs, so <u,1>_boundary = u^T M_facet 1.
    la::MatrixCSR<double> Mf(pattern);
    fem::assemble_matrix(Mf.mat_add_values(), make_form({V, V}, mesh,
        fem::IntegralType::exterior_facet, k_facet, facet_entities),
        std::vector<std::reference_wrapper<const fem::DirichletBC<double>>> {});
    // u^T (Mf 1) = sum_i u_i (row sum of Mf).
    double u_bc = 0;
    const auto& rp = Mf.row_ptr();
    for (std::size_t i = 0; i < u.array().size(); ++i) {
        double row_sum = 0;
        for (std::int64_t k = rp[i]; k < rp[i + 1]; ++k)
            row_sum += Mf.values()[static_cast<std::size_t>(k)];
        u_bc += u.array()[i] * row_sum;
    }
    REQUIRE(u_bc == Approx(1.0).margin(1e-8));
}

TEST_CASE("facet: interior facet jump term is symmetric with zero row sums", "[fem]")
{
    // DG-style: <jump(u), jump(v)> over interior facets. Constant
    // functions have zero jump, so the row sums vanish and the matrix is
    // symmetric.
    auto V = p1_space(2);
    auto mesh = V->mesh();
    auto coord = mesh->geometry().cmaps().front();
    fem::FacetPrecomputeData<double> pre(mesh::CellType::triangle,
        *V->element(), *V->element(), {}, coord, 2);
    auto k_jump = fem::make_interior_facet_kernel(pre, jump_kernel);
    auto interior = fem::interior_facet_entities(*mesh->topology());
    REQUIRE_FALSE(interior.empty());

    auto a = make_form({V, V}, mesh, fem::IntegralType::interior_facet,
        k_jump, interior);

    auto cells = all_cells(*V->dofmap());
    la::SparsityPattern pattern(V->dofmap()->index_map, 1);
    fem::sparsitybuild::cells(pattern, std::pair {cells, cells},
        {*V->dofmap(), *V->dofmap()});
    // Interior facets couple both cells on each side; the sparsity builder
    // expects interleaved (cell0, cell1) pairs per facet.
    std::vector<std::int32_t> side_cells;
    for (std::size_t f = 0; f < interior.size() / 4; ++f) {
        side_cells.push_back(interior[4 * f]);
        side_cells.push_back(interior[4 * f + 2]);
    }
    std::array<std::span<const std::int32_t>, 2> facet_cells {
        std::span(side_cells), std::span(side_cells)};
    fem::sparsitybuild::interior_facets(pattern, facet_cells,
        {*V->dofmap(), *V->dofmap()});
    pattern.finalize();

    la::MatrixCSR<double> A(pattern);
    fem::assemble_matrix(A.mat_add_values(), a, std::vector<std::reference_wrapper<const fem::DirichletBC<double>>>{});

    auto dense = A.to_dense();
    const std::size_t N = static_cast<std::size_t>(V->dofmap()->index_map->size_local());
    for (std::size_t i = 0; i < N; ++i) {
        double row_sum = 0;
        for (std::size_t j = 0; j < N; ++j) {
            REQUIRE(dense[i * N + j] == Approx(dense[j * N + i]).margin(1e-10));
            row_sum += dense[i * N + j];
        }
        REQUIRE(row_sum == Approx(0.0).margin(1e-10));
    }
}
