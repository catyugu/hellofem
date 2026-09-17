// hellofem::fem — interpolation tests
// SPDX-License-Identifier: MIT

#include "basis/element-families.h"
#include "basis/finite-element.h"
#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "common/IndexMap.h"
#include "fem/CoordinateElement.h"
#include "fem/DofMap.h"
#include "fem/FiniteElement.h"
#include "fem/Function.h"
#include "fem/FunctionSpace.h"
#include "fem/dofmapbuilder.h"
#include "fem/utils.h"
#include "mesh/Mesh.h"
#include "mesh/Topology.h"
#include "mesh/cell_types.h"
#include "mesh/generation.h"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <map>
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

} // namespace

TEST_CASE("interpolate: P1 projection of a quadratic field", "[fem]")
{
    auto V = p1_space(2);
    fem::Function<double> u(V);

    // Interpolate f(x,y) = x^2 + y^2. For P1, nodal interpolation is
    // exact at the dof coordinates (the field is evaluated at the
    // interpolation points, which coincide with the mesh vertices).
    u.interpolate(
        [](std::span<const double> X, std::array<std::size_t, 2> shape) {
            const std::size_t n = shape[0];
            std::vector<double> f(n);
            for (std::size_t i = 0; i < n; ++i)
                f[i] = X[2 * i] * X[2 * i] + X[2 * i + 1] * X[2 * i + 1];
            return std::make_pair(std::move(f),
                std::array<std::size_t, 2> {n, 1});
        });

    // At each dof, the interpolated value equals f at the dof coordinate.
    auto coords = V->tabulate_dof_coordinates(false);
    for (std::int32_t d = 0; d < V->dofmap()->index_map->size_local(); ++d) {
        const double x = coords[2 * d];
        const double y = coords[2 * d + 1];
        REQUIRE(u.x()->array()[static_cast<std::size_t>(d)]
            == Approx(x * x + y * y).margin(1e-12));
    }
}

TEST_CASE("interpolate: copy of a function on the same space", "[fem]")
{
    auto V = p1_space(1);
    fem::Function<double> u(V);
    u.interpolate(
        [](std::span<const double> X, std::array<std::size_t, 2> shape) {
            const std::size_t n = shape[0];
            std::vector<double> f(n);
            for (std::size_t i = 0; i < n; ++i)
                f[i] = 3.0 * X[2 * i] + 1.0;
            return std::make_pair(std::move(f),
                std::array<std::size_t, 2> {n, 1});
        });

    fem::Function<double> v(V);
    v.interpolate(u);
    for (std::size_t i = 0; i < u.x()->array().size(); ++i)
        REQUIRE(v.x()->array()[i] == Approx(u.x()->array()[i]));
}

namespace {

    /// Two tetrahedra sharing the face (1,2,4): the two cells traverse the
    /// edges of that face in opposite directions, which is what an element's
    /// dof permutation has to reconcile. They lie on opposite sides of the
    /// shared face, so together they are a valid mesh.
    std::shared_ptr<mesh::Mesh<double>> tet_box()
    {
        const std::vector<double> x {0, 0, 0, /**/ 1, 0, 0, /**/ 0, 1, 0,
            /**/ 1, 1, 0, /**/ 0, 0, 1};
        const std::vector<std::int64_t> cells {0, 1, 2, 4, /**/ 2, 1, 3, 4};
        return std::make_shared<mesh::Mesh<double>>(mesh::create_mesh(
            std::span<const std::int64_t>(cells),
            mesh::CellType::tetrahedron, x, 3));
    }

    /// Build a P`order` FunctionSpace on a tetrahedral box.
    std::shared_ptr<fem::FunctionSpace<double>> box_space(int order, int n)
    {
        (void)n; // one fixture, kept for the call sites' shape
        auto m = tet_box();
        auto fe = std::make_shared<fem::FiniteElement<double>>(
            basis::create_element<double>(
                B::P, C::tetrahedron, order, LV::equispaced, DV::unset, false));
        return fem::create_functionspace(m, std::move(fe));
    }

    /// A polynomial of every monomial up to `degree`, so a P`degree` space
    /// must reproduce it exactly and a higher-order one trivially so.
    double polynomial(std::span<const double> p, int degree)
    {
        const double x = p[0], y = p[1], z = p[2];
        double v = 0;
        const double xs[5] = {1.0, x, x * x, x * x * x, x * x * x * x};
        const double ys[5] = {1.0, y, y * y, y * y * y, y * y * y * y};
        const double zs[5] = {1.0, z, z * z, z * z * z, z * z * z * z};
        for (int i = 0; i <= degree; ++i)
            for (int j = 0; i + j <= degree; ++j)
                for (int k = 0; i + j + k <= degree; ++k)
                    v += (1.0 + i + 2.0 * j + 3.0 * k) * xs[i] * ys[j] * zs[k];
        return v;
    }

} // namespace

TEST_CASE("interpolate: a Lagrange space reproduces its own degree exactly",
    "[fem]")
{
    // A nodal interpolation is exact at the dofs by construction, so the
    // test is the VALUE BETWEEN them: a wrong dof layout, or a basis whose
    // points do not match it, reproduces the polynomial at the dofs and
    // fails everywhere else. On several cells the dofs of a shared entity
    // must also agree about which point they name.
    // Interior points of both cells: four in the corner tetrahedron and
    // four in its neighbour across the shared face.
    const std::vector<double> points {0.25, 0.25, 0.25, 0.1, 0.1, 0.1, 0.6, 0.2,
        0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.25, 0.4, 0.4, 0.3, 0.5, 0.3, 0.3, 0.3,
        0.35, 0.4};
    for (const int order : {1, 2, 3, 4}) {
        auto V = box_space(order, 0);
        fem::Function<double> u(V);
        u.interpolate([order](std::span<const double> X,
                          std::array<std::size_t, 2> shape) {
            const std::size_t n = shape[0];
            std::vector<double> f(n);
            for (std::size_t i = 0; i < n; ++i)
                f[i] = polynomial(X.subspan(3 * i, 3), order);
            return std::make_pair(std::move(f),
                std::array<std::size_t, 2> {n, 1});
        });

        const std::size_t np = points.size() / 3;
        auto [values, vshape] = u.eval(points, {np, 3});
        for (std::size_t i = 0; i < np; ++i) {
            const double want = polynomial(
                std::span<const double>(points.data() + 3 * i, 3), order);
            if (std::abs(values[i] - want) > 1e-9)
                std::fprintf(stderr,
                    "MISMATCH order=%d point=%zu "
                    "at (%.2f,%.2f,%.2f) got=%.6f want=%.6f\n",
                    order, i, points[3 * i], points[3 * i + 1],
                    points[3 * i + 2], values[i], want);
            INFO("order " << order << " point " << i);
            REQUIRE(values[i] == Approx(want).margin(1e-9));
        }
    }
}
