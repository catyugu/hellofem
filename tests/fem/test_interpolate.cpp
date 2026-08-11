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
