// hellofem::fem — Constant and Function unit tests
// SPDX-License-Identifier: MIT

#include "basis/element-families.h"
#include "basis/finite-element.h"
#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "common/IndexMap.h"
#include "fem/Constant.h"
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

#include <cstdint>
#include <memory>
#include <vector>

using namespace hellofem;
using Catch::Approx;

namespace {

    using B = basis::element::family;
    using C = basis::cell::type;
    using LV = basis::element::lagrange_variant;
    using DV = basis::element::dpc_variant;

    fem::ElementDofLayout p1_layout()
    {
        return fem::CoordinateElement<double>(
            mesh::CellType::triangle, 1, LV::equispaced)
            .create_dof_layout();
    }

    std::shared_ptr<fem::FunctionSpace<double>> p1_space()
    {
        auto mesh = mesh::create_unit_square(1);
        auto fe = std::make_shared<fem::FiniteElement<double>>(
            basis::create_element<double>(
                B::P, C::triangle, 1, LV::equispaced, DV::unset, false));
        auto [imap, bs, dofmaps]
            = fem::build_dofmap_data(*mesh->topology(), {p1_layout()}, nullptr);
        auto dmap = std::make_shared<fem::DofMap>(p1_layout(),
            std::make_shared<common::IndexMap>(std::move(imap)), bs,
            std::move(dofmaps.front()), bs);
        return std::make_shared<fem::FunctionSpace<double>>(mesh, fe, dmap);
    }

} // namespace

TEST_CASE("Constant: scalar and vector constants", "[fem]")
{
    fem::Constant<double> c0(3.0);
    REQUIRE(c0.value.size() == 1);
    REQUIRE(c0.value[0] == Approx(3.0));
    REQUIRE(c0.shape.empty());

    std::vector<double> v {1.0, 2.0, 3.0};
    fem::Constant<double> c1(v);
    REQUIRE(c1.value.size() == 3);
    REQUIRE(c1.shape.size() == 1);
    REQUIRE(c1.shape[0] == 3);
    REQUIRE(c1.value[2] == Approx(3.0));
}

TEST_CASE("Function: allocates a vector sized by the space", "[fem]")
{
    auto V = p1_space();
    fem::Function<double> u(V);
    REQUIRE(u.function_space() == V);
    REQUIRE(u.x()->array().size() == 4);
    // Fresh Function is zero.
    for (double v : u.x()->array())
        REQUIRE(v == Approx(0.0));

    u.x()->array()[1] = 2.5;
    REQUIRE(u.x()->array()[1] == Approx(2.5));
}

TEST_CASE("Function: interpolate from point values at interpolation points",
    "[fem]")
{
    auto V = p1_space();
    fem::Function<double> u(V);

    // Interpolate f(x, y) = x + 2y into a P1 function: the dof values
    // equal the field values at the 4 corners.
    u.interpolate(
        [](std::span<const double> X,
            std::array<std::size_t, 2> shape) {
            const std::size_t n = shape[0];
            std::vector<double> f(n);
            for (std::size_t i = 0; i < n; ++i)
                f[i] = X[2 * i + 0] + 2 * X[2 * i + 1];
            return std::make_pair(std::move(f),
                std::array<std::size_t, 2> {n, 1});
        });

    REQUIRE(u.x()->array()[0] == Approx(0.0)); // corner (0, 0)
    REQUIRE(u.x()->array()[1] == Approx(1.0)); // corner (1, 0)
    REQUIRE(u.x()->array()[2] == Approx(2.0)); // corner (0, 1)
    REQUIRE(u.x()->array()[3] == Approx(3.0)); // corner (1, 1)
}
