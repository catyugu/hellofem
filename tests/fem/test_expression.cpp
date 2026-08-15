// hellofem::fem — Expression and tabulate_expression tests
// SPDX-License-Identifier: MIT

#include "basis/element-families.h"
#include "basis/finite-element.h"
#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "common/IndexMap.h"
#include "fem/Constant.h"
#include "fem/CoordinateElement.h"
#include "fem/DofMap.h"
#include "fem/Expression.h"
#include "fem/FiniteElement.h"
#include "fem/Function.h"
#include "fem/FunctionSpace.h"
#include "fem/dofmapbuilder.h"
#include "fem/precompute.h"
#include "fem/tabulate_expression.h"
#include "mesh/Mesh.h"
#include "mesh/Topology.h"
#include "mesh/cell_types.h"
#include "mesh/generation.h"

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
        auto layout = fe->create_dof_layout();
        auto [imap, bs, dofmaps]
            = fem::build_dofmap_data(*mesh->topology(), {layout}, nullptr);
        auto dmap = std::make_shared<fem::DofMap>(layout,
            std::make_shared<common::IndexMap>(std::move(imap)), bs,
            std::move(dofmaps.front()), bs);
        return std::make_shared<fem::FunctionSpace<double>>(mesh, fe, dmap);
    }

    /// A writer that outputs the physical coordinate (x, y) of each
    /// quadrature point.
    void write_coords(double* out, const fem::CellKernelData<double>& data)
    {
        for (int p = 0; p < data.num_points; ++p) {
            out[2 * p + 0] = data.X[p * 3 + 0];
            out[2 * p + 1] = data.X[p * 3 + 1];
        }
    }

} // namespace

TEST_CASE("expression: physical coordinates at quadrature points", "[fem]")
{
    auto V = p1_space(1); // 2 triangles on the unit square.

    // Precompute a 1-point quadrature rule (degree 0) with an empty
    // test/trial (use P1 as the space placeholder; coefficients unused).
    auto coord_el = fem::CoordinateElement<double>(
        mesh::CellType::triangle, 1, LV::equispaced);
    auto pre = std::make_shared<fem::PrecomputeData<double>>(
        mesh::CellType::triangle, *V->element(), *V->element(),
        std::vector<const fem::FiniteElement<double>*> {}, coord_el, 0);

    auto kernel = fem::make_expression_kernel(*pre, write_coords);
    const auto nq = pre->points().size() / 2; // (nq, tdim)
    fem::Expression<double> expr(
        {}, {std::make_shared<const fem::Constant<double>>(0.0)},
        pre->points(), {nq, 2}, std::move(kernel),
        std::vector<std::size_t> {2});

    // Evaluate over the two cells. Cell 0 is the lower-left triangle
    // (vertices (0,0),(1,0),(0,1)); cell 1 the upper-right.
    const auto x_dofmap = V->mesh()->geometry().dofmaps().front();
    std::span<const double> xg = V->mesh()->geometry().x();

    // The quadrature point of the reference triangle at degree 0 is the
    // centroid (1/3, 1/3). Push it forward per cell.
    std::vector<std::int32_t> cells {0, 1};
    auto values = fem::tabulate_expression(expr, *V->mesh(), cells);

    // Cell 0 centroid = (1/3, 1/3), cell 1 centroid = (2/3, 2/3).
    std::array<double, 2> expect = {1.0 / 3.0, 2.0 / 3.0};
    for (int c = 0; c < 2; ++c) {
        const double cx = values[2 * c + 0];
        const double cy = values[2 * c + 1];
        REQUIRE(cx == Approx(expect[static_cast<std::size_t>(c)]).margin(1e-12));
        REQUIRE(cy == Approx(expect[static_cast<std::size_t>(c)]).margin(1e-12));
    }
}

TEST_CASE("expression: depends on a coefficient function", "[fem]")
{
    auto V = p1_space(2);
    fem::Function<double> u(V);
    u.interpolate(
        [](std::span<const double> X, std::array<std::size_t, 2> shape) {
            const std::size_t n = shape[0];
            std::vector<double> f(n);
            for (std::size_t i = 0; i < n; ++i)
                f[i] = X[2 * i] + 2.0 * X[2 * i + 1];
            return std::make_pair(std::move(f),
                std::array<std::size_t, 2> {n, 1});
        });

    // Expression: 2*u. Writer outputs data.coeffs[0][p].
    auto pre = std::make_shared<fem::PrecomputeData<double>>(
        mesh::CellType::triangle, *V->element(), *V->element(),
        std::vector<const fem::FiniteElement<double>*> {V->element().get()},
        fem::CoordinateElement<double>(mesh::CellType::triangle, 1, LV::equispaced),
        0);
    auto kernel = fem::make_expression_kernel(*pre,
        [](double* out, const fem::CellKernelData<double>& data) {
            for (int p = 0; p < data.num_points; ++p)
                out[p] = 2.0 * data.coeffs[p];
        });
    const auto nq = pre->points().size() / 2;
    fem::Expression<double> expr(
        {std::make_shared<const fem::Function<double>>(u)}, {},
        pre->points(), {nq, 2}, std::move(kernel),
        std::vector<std::size_t> {});

    // Cell 0 of the n=2 mesh has vertices (0,0), (0.5,0), (0,0.5); its
    // physical centroid is (1/6, 1/6) where u = 1/6 + 2/6 = 0.5, so the
    // expression 2*u = 1.0.
    std::vector<std::int32_t> cells {0};
    auto values = fem::tabulate_expression(expr, *V->mesh(), cells);
    REQUIRE(values[0] == Approx(1.0).margin(1e-12));
}

TEST_CASE("expression: gradient of a coefficient (Joule-heating style)", "[fem]")
{
    auto V = p1_space(2);
    // u(x,y) = x^2 (degree 2, represented exactly by P2).
    // grad u = (2x, 0); |grad u|^2 = 4 x^2.
    const auto ufield = [](std::span<const double> X,
                             std::array<std::size_t, 2> shape) {
        const std::size_t n = shape[0];
        std::vector<double> f(n);
        for (std::size_t i = 0; i < n; ++i)
            f[i] = X[2 * i] * X[2 * i];
        return std::make_pair(std::move(f), std::array<std::size_t, 2> {n, 1});
    };

    auto fe2 = std::make_shared<fem::FiniteElement<double>>(
        basis::create_element<double>(
            B::P, C::triangle, 2, LV::equispaced, DV::unset, false));
    auto layout2 = fe2->create_dof_layout();
    auto* topology = V->mesh()->topology_mutable().get();
    for (int d = 1; d < topology->dim(); ++d)
        topology->create_entities(d);
    auto [imap2, bs2, dofmaps2]
        = fem::build_dofmap_data(*topology, {layout2}, nullptr);
    auto dmap2 = std::make_shared<fem::DofMap>(layout2,
        std::make_shared<common::IndexMap>(std::move(imap2)), bs2,
        std::move(dofmaps2.front()), bs2);
    auto V2 = std::make_shared<fem::FunctionSpace<double>>(V->mesh(), fe2, dmap2);
    fem::Function<double> u2(V2);
    u2.interpolate(ufield);

    auto pre = std::make_shared<fem::PrecomputeData<double>>(
        mesh::CellType::triangle, *V2->element(), *V2->element(),
        std::vector<const fem::FiniteElement<double>*> {V2->element().get()},
        fem::CoordinateElement<double>(mesh::CellType::triangle, 1, LV::equispaced),
        0);
    auto kernel = fem::make_expression_kernel(*pre,
        [](double* out, const fem::CellKernelData<double>& data) {
            // |grad u|^2 = dcoeffs[0*3+0]^2 + dcoeffs[0*3+1]^2
            for (int p = 0; p < data.num_points; ++p) {
                const double gx = data.dcoeffs[p * 3 + 0];
                const double gy = data.dcoeffs[p * 3 + 1];
                out[p] = gx * gx + gy * gy;
            }
        });
    const auto nq = pre->points().size() / 2;
    fem::Expression<double> expr(
        {std::make_shared<const fem::Function<double>>(u2)}, {},
        pre->points(), {nq, 2}, std::move(kernel),
        std::vector<std::size_t> {});

    // Cell 0 of the n=2 mesh: physical centroid (1/6, 1/6), where
    // |grad u|^2 = 4 x^2 = 4/36 = 1/9.
    std::vector<std::int32_t> cells {0};
    auto values = fem::tabulate_expression(expr, *V2->mesh(), cells);
    REQUIRE(values[0] == Approx(1.0 / 9.0).margin(1e-9));
}
