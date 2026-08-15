// hellofem::fem — Function::eval point-evaluation tests
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
#include <stdexcept>
#include <vector>

using namespace hellofem;
using Catch::Approx;

namespace {

    using B = basis::element::family;
    using C = basis::cell::type;
    using LV = basis::element::lagrange_variant;
    using DV = basis::element::dpc_variant;

    /// Build a FunctionSpace of given degree on an n-interval unit square.
    std::shared_ptr<fem::FunctionSpace<double>> p_space(int n, int degree)
    {
        auto mesh = mesh::create_unit_square(n);
        auto fe = std::make_shared<fem::FiniteElement<double>>(
            basis::create_element<double>(
                B::P, C::triangle, degree, LV::equispaced, DV::unset, false));
        auto layout = fe->create_dof_layout();
        // Dofmaps with entity dofs (e.g. P2 edges) require the mesh
        // entities to exist before the dofmap is built.
        auto* topology = mesh->topology_mutable().get();
        for (int d = 1; d < topology->dim(); ++d)
            topology->create_entities(d);
        auto [imap, bs, dofmaps]
            = fem::build_dofmap_data(*topology, {layout}, nullptr);
        auto dmap = std::make_shared<fem::DofMap>(layout,
            std::make_shared<common::IndexMap>(std::move(imap)), bs,
            std::move(dofmaps.front()), bs);
        return std::make_shared<fem::FunctionSpace<double>>(mesh, fe, dmap);
    }

} // namespace

TEST_CASE("eval: P2 field at dof coordinates matches the interpolated values",
    "[fem]")
{
    // Interpolate a smooth field on P2; the function exactly represents it.
    // Evaluate at the dof coordinates and compare to the exact field.
    auto V = p_space(4, 2);
    fem::Function<double> u(V);
    u.interpolate(
        [](std::span<const double> X, std::array<std::size_t, 2> shape) {
            const std::size_t n = shape[0];
            std::vector<double> f(n);
            for (std::size_t i = 0; i < n; ++i)
                f[i] = std::sin(X[2 * i]) * std::cos(X[2 * i + 1]);
            return std::make_pair(std::move(f),
                std::array<std::size_t, 2> {n, 1});
        });

    auto coords = V->tabulate_dof_coordinates(false);
    const std::size_t num_dofs = coords.size() / 2;
    std::vector<double> x(2 * num_dofs);
    std::vector<std::int32_t> cells(num_dofs, -1);
    for (std::size_t d = 0; d < num_dofs; ++d) {
        x[2 * d] = coords[2 * d];
        x[2 * d + 1] = coords[2 * d + 1];
    }
    // Locate the cell of each dof coordinate.
    for (std::size_t c = 0; c < V->dofmap()->map().extent(0); ++c) {
        auto dofs = V->dofmap()->cell_dofs(static_cast<std::int32_t>(c));
        for (std::int32_t d : dofs)
            cells[static_cast<std::size_t>(d)] = static_cast<std::int32_t>(c);
    }

    std::vector<double> uval(num_dofs, 0);
    u.eval(x, {num_dofs, 2}, cells, uval, {num_dofs, 1});
    for (std::size_t d = 0; d < num_dofs; ++d) {
        const double X = coords[2 * d];
        const double Y = coords[2 * d + 1];
        REQUIRE(uval[d] == Approx(std::sin(X) * std::cos(Y)).margin(1e-9));
    }
}

TEST_CASE("eval: blocked vector field interleaves components", "[fem]")
{
    // Vector P1 (bs = 2) with field (x, y): each component is reproduced.
    auto mesh = mesh::create_unit_square(2);
    auto fe = std::make_shared<fem::FiniteElement<double>>(
        basis::create_element<double>(
            B::P, C::triangle, 1, LV::equispaced, DV::unset, false),
        std::vector<std::size_t> {2});
    auto layout = fe->create_dof_layout();
    auto* topology = mesh->topology_mutable().get();
    for (int d = 1; d < topology->dim(); ++d)
        topology->create_entities(d);
    auto [imap, bs, dofmaps]
        = fem::build_dofmap_data(*topology, {layout}, nullptr);
    auto dmap = std::make_shared<fem::DofMap>(layout,
        std::make_shared<common::IndexMap>(std::move(imap)), bs,
        std::move(dofmaps.front()), bs);
    auto V = std::make_shared<fem::FunctionSpace<double>>(mesh, fe, dmap);
    REQUIRE(bs == 2);

    fem::Function<double> u(V);
    // Set the coefficients directly to the field (x, y) at each vertex.
    // For a blocked P1 element each dof block is a mesh vertex; physical
    // dof index = bs * vertex + component.
    std::span<const double> xg = mesh->geometry().x();
    for (std::size_t c = 0; c < dmap->map().extent(0); ++c) {
        auto dofs = dmap->cell_dofs(static_cast<std::int32_t>(c));
        for (std::int32_t v : dofs) {
            u.x()->array()[static_cast<std::size_t>(2 * v)]
                = xg[static_cast<std::size_t>(3 * v)];
            u.x()->array()[static_cast<std::size_t>(2 * v + 1)]
                = xg[static_cast<std::size_t>(3 * v + 1)];
        }
    }

    // Evaluate at the mesh vertices via the automatic cell-location
    // overload and check each component.
    const std::int32_t num_vertices
        = mesh->topology()->index_map(0)->size_local();
    std::vector<double> x(2 * static_cast<std::size_t>(num_vertices));
    for (std::int32_t v = 0; v < num_vertices; ++v) {
        x[2 * static_cast<std::size_t>(v)] = xg[3 * static_cast<std::size_t>(v)];
        x[2 * static_cast<std::size_t>(v) + 1]
            = xg[3 * static_cast<std::size_t>(v) + 1];
    }
    auto [uval, ushape] = u.eval(x, {
        static_cast<std::size_t>(num_vertices), 2});
    REQUIRE(ushape == std::array<std::size_t, 2> {
        static_cast<std::size_t>(num_vertices), 2});
    for (std::int32_t v = 0; v < num_vertices; ++v) {
        REQUIRE(uval[2 * static_cast<std::size_t>(v)]
            == Approx(x[2 * static_cast<std::size_t>(v)]).margin(1e-12));
        REQUIRE(uval[2 * static_cast<std::size_t>(v) + 1]
            == Approx(x[2 * static_cast<std::size_t>(v) + 1]).margin(1e-12));
    }
}

TEST_CASE("eval: mixed element is rejected", "[fem]")
{
    auto V = p_space(2, 1);
    fem::Function<double> u(V);

    // Build a mixed element (two distinct sub-elements).
    auto e1 = basis::create_element<double>(
        B::P, C::triangle, 1, LV::equispaced, DV::unset, false);
    auto e2 = basis::create_element<double>(
        B::P, C::triangle, 2, LV::equispaced, DV::unset, false);
    auto mixed = std::make_shared<fem::FiniteElement<double>>(
        std::vector<std::shared_ptr<const fem::FiniteElement<double>>> {
            std::make_shared<fem::FiniteElement<double>>(e1),
            std::make_shared<fem::FiniteElement<double>>(e2)});
    auto layout = fem::CoordinateElement<double>(
        mesh::CellType::triangle, 1, LV::equispaced)
                      .create_dof_layout();
    auto* topology = V->mesh()->topology_mutable().get();
    for (int d = 1; d < topology->dim(); ++d)
        topology->create_entities(d);
    auto [imap, bs, dofmaps]
        = fem::build_dofmap_data(*topology, {layout}, nullptr);
    auto dmap = std::make_shared<fem::DofMap>(layout,
        std::make_shared<common::IndexMap>(std::move(imap)), bs,
        std::move(dofmaps.front()), bs);
    auto Vm = std::make_shared<fem::FunctionSpace<double>>(V->mesh(), mixed, dmap);
    fem::Function<double> w(Vm);

    std::vector<double> x(2, 0);
    std::vector<std::int32_t> cells(1, 0);
    std::vector<double> uval(2, 0);
    REQUIRE_THROWS(w.eval(x, {1, 2}, cells, uval, {1, 2}));
}

TEST_CASE("eval: convenience overload locates cells automatically", "[fem]")
{
    auto V = p_space(3, 2);
    fem::Function<double> u(V);
    u.interpolate(
        [](std::span<const double> X, std::array<std::size_t, 2> shape) {
            const std::size_t n = shape[0];
            std::vector<double> f(n);
            for (std::size_t i = 0; i < n; ++i)
                f[i] = X[2 * i] * X[2 * i] + X[2 * i + 1] * X[2 * i + 1];
            return std::make_pair(std::move(f),
                std::array<std::size_t, 2> {n, 1});
        });

    // Points on a grid inside the unit square.
    constexpr std::size_t m = 4;
    std::vector<double> x(2 * m * m);
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < m; ++j) {
            x[2 * (i * m + j)] = 0.1 + 0.25 * static_cast<double>(i);
            x[2 * (i * m + j) + 1] = 0.1 + 0.25 * static_cast<double>(j);
        }

    auto [values, shape] = u.eval(x, {m * m, 2});
    REQUIRE(shape == std::array<std::size_t, 2> {m * m, 1});
    for (std::size_t p = 0; p < m * m; ++p) {
        const double X = x[2 * p];
        const double Y = x[2 * p + 1];
        REQUIRE(values[p] == Approx(X * X + Y * Y).margin(1e-9));
    }
}
