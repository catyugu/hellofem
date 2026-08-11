// hellofem::fem — Form assembly engine tests
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
#include "fem/Function.h"
#include "fem/FunctionSpace.h"
#include "fem/assembler.h"
#include "fem/dofmapbuilder.h"
#include "fem/precompute.h"
#include "fem/sparsitybuild.h"
#include "la/MatrixCSR.h"
#include "la/SparsityPattern.h"
#include "la/Vector.h"
#include "mesh/Mesh.h"
#include "mesh/Topology.h"
#include "mesh/generation.h"

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

    /// Build a P1 FunctionSpace on a unit square.
    std::shared_ptr<fem::FunctionSpace<double>>
    p1_space(int n = 1)
    {
        std::fprintf(stderr, "P1SPACE: after create_unit_square\n");
        auto mesh = mesh::create_unit_square(n);
        std::fprintf(stderr, "P1SPACE: after create_unit_square returns\n");
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

    /// A P1 poisson mass kernel: Ae[i,j] = sum_q w_q * detJ * phi_i phi_j.
    void mass_kernel(double* Ae, const fem::CellKernelData<double>& data)
    {
        const int nq = data.num_points;
        const int nd = data.num_dofs0;
        std::fill(Ae, Ae + nd * nd, 0);
        for (int q = 0; q < nq; ++q)
            for (int i = 0; i < nd; ++i)
                for (int j = 0; j < nd; ++j)
                    Ae[i * nd + j] += data.w[q] * data.detJ[q]
                        * data.phi0[q * nd + i] * data.phi1[q * nd + j];
    }

    /// A P1 poisson stiffness kernel: Ae[i,j] = sum_q w_q * detJ *
    /// grad(phi_i) . grad(phi_j).
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

} // namespace

TEST_CASE("Form: mass matrix of P1 on 2-triangle unit square", "[fem]")
{
    auto V = p1_space(1);
    auto mesh = V->mesh();
    auto coord = mesh->geometry().cmaps().front();

    fem::PrecomputeData<double> pre(mesh::CellType::triangle, *V->element(),
        *V->element(), {}, coord, 2);
    fem::Form<double>::integral_data cell_integral;
    cell_integral.kernel = fem::make_cell_kernel(pre, mass_kernel);
    std::fprintf(stderr, "TEST: kernel created\n");
    cell_integral.entities = {0, 1};
    cell_integral.coeffs = {};

    std::vector<std::shared_ptr<const fem::FunctionSpace<double>>> Vlist {V, V};
    std::map<std::pair<fem::IntegralType, int>,
        std::vector<fem::Form<double>::integral_data>>
        integrals;
    integrals[{fem::IntegralType::cell, 0}] = {cell_integral};
    fem::Form<double> M(Vlist, std::move(integrals), mesh, {}, {});
    std::fprintf(stderr, "TEST: form built\n");

    // Sparsity: P1 on 2 tris shares the diagonal -> 4 dofs, all-to-all.
    la::SparsityPattern pattern(V->dofmap()->index_map, 1);
    fem::sparsitybuild::cells(pattern,
        std::pair {std::vector<std::int32_t> {0, 1}, std::vector<std::int32_t> {0, 1}},
        {*V->dofmap(), *V->dofmap()});
    pattern.insert_diagonal(std::span<const std::int32_t>(std::vector<std::int32_t> {0, 1, 2, 3}));
    pattern.finalize();
    std::fprintf(stderr, "TEST: pattern built\n");

    la::MatrixCSR<double> A(pattern);
    std::fprintf(stderr, "TEST: matrix built\n");
    fem::assemble_matrix(A.mat_add_values(), M, std::span<const std::int8_t> {},
        std::span<const std::int8_t> {});
    std::fprintf(stderr, "TEST: assembled\n");

    auto K = A.to_dense();
    // Known mass matrix for two unit right triangles sharing a diagonal:
    // vertices (0,0),(1,0),(0,1),(1,1), cells {0,1,2} and {1,3,2}.
    // Each unit triangle (area 1/2) has M_ij = area/12 * (1 + delta_ij).
    // Vertices 1, 2 are shared by both triangles; 0 and 3 by one.
    const int n = 4;
    REQUIRE(K.size() == static_cast<std::size_t>(n * n));
    REQUIRE(K[0 * n + 0] == Approx(1.0 / 12.0));
    REQUIRE(K[1 * n + 1] == Approx(1.0 / 6.0));
    REQUIRE(K[2 * n + 2] == Approx(1.0 / 6.0));
    REQUIRE(K[3 * n + 3] == Approx(1.0 / 12.0));
    // Shared edge (1,2) contributes from both triangles.
    REQUIRE(K[1 * n + 2] == Approx(1.0 / 12.0));
    REQUIRE(K[1 * n + 3] == Approx(1.0 / 24.0));
    REQUIRE(K[2 * n + 3] == Approx(1.0 / 24.0));
    REQUIRE(K[0 * n + 3] == Approx(0.0)); // no shared triangle
    // Symmetry.
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            REQUIRE(K[i * n + j] == Approx(K[j * n + i]));
}

TEST_CASE("Form: stiffness matrix of P1 on 2-triangle unit square", "[fem]")
{
    auto V = p1_space(1);
    auto mesh = V->mesh();
    auto coord = mesh->geometry().cmaps().front();

    fem::PrecomputeData<double> pre(mesh::CellType::triangle, *V->element(),
        *V->element(), {}, coord, 2);
    fem::Form<double>::integral_data cell_integral;
    cell_integral.kernel = fem::make_cell_kernel(pre, stiffness_kernel);
    cell_integral.entities = {0, 1};
    cell_integral.coeffs = {};

    std::vector<std::shared_ptr<const fem::FunctionSpace<double>>> Vlist {V, V};
    std::map<std::pair<fem::IntegralType, int>,
        std::vector<fem::Form<double>::integral_data>>
        integrals;
    integrals[{fem::IntegralType::cell, 0}] = {cell_integral};
    fem::Form<double> A_form(Vlist, std::move(integrals), mesh, {}, {});

    la::SparsityPattern pattern(V->dofmap()->index_map, 1);
    fem::sparsitybuild::cells(pattern,
        std::pair {std::vector<std::int32_t> {0, 1}, std::vector<std::int32_t> {0, 1}},
        {*V->dofmap(), *V->dofmap()});
    pattern.insert_diagonal(std::span<const std::int32_t>(std::vector<std::int32_t> {0, 1, 2, 3}));
    pattern.finalize();

    la::MatrixCSR<double> A(pattern);
    fem::assemble_matrix(A.mat_add_values(), A_form,
        std::span<const std::int8_t> {}, std::span<const std::int8_t> {});

    auto K = A.to_dense();
    // Known P1 stiffness matrix for the two unit right triangles
    // (cells {0,1,2} and {1,3,2}) — the classic 4x4:
    //   [ 1   -.5  -.5  0 ]
    //   [-.5   1   0   -.5]
    //   [-.5   0   1   -.5]
    //   [ 0   -.5  -.5  1 ]
    const int n = 4;
    REQUIRE(K[0 * n + 0] == Approx(1.0));
    REQUIRE(K[0 * n + 1] == Approx(-0.5));
    REQUIRE(K[0 * n + 2] == Approx(-0.5));
    REQUIRE(K[0 * n + 3] == Approx(0.0));
    REQUIRE(K[1 * n + 1] == Approx(1.0));
    REQUIRE(K[1 * n + 2] == Approx(0.0));
    REQUIRE(K[1 * n + 3] == Approx(-0.5));
    REQUIRE(K[2 * n + 2] == Approx(1.0));
    REQUIRE(K[2 * n + 3] == Approx(-0.5));
    REQUIRE(K[3 * n + 3] == Approx(1.0));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            REQUIRE(K[i * n + j] == Approx(K[j * n + i]));
}
