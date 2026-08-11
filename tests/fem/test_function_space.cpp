// hellofem::fem — FunctionSpace unit tests
// SPDX-License-Identifier: MIT

#include "basis/element-families.h"
#include "basis/finite-element.h"
#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "common/IndexMap.h"
#include "fem/CoordinateElement.h"
#include "fem/DofMap.h"
#include "fem/FiniteElement.h"
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

namespace {

    using B = basis::element::family;
    using C = basis::cell::type;
    using LV = basis::element::lagrange_variant;
    using DV = basis::element::dpc_variant;

    /// Scalar P1 dof layout on a triangle.
    fem::ElementDofLayout p1_layout()
    {
        return fem::CoordinateElement<double>(
            mesh::CellType::triangle, 1, LV::equispaced)
            .create_dof_layout();
    }

    /// Build a FunctionSpace on a single P1 unit square (2 triangles).
    std::shared_ptr<fem::FunctionSpace<double>> p1_unit_square_space()
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

TEST_CASE("FunctionSpace: P1 on a unit square (2 triangles)", "[fem]")
{
    auto V = p1_unit_square_space();
    // Local (reference-cell) dimension is 3; the global space has 4 dofs.
    REQUIRE(V->element()->space_dimension() == 3);
    REQUIRE(V->dofmap()->index_map->size_local() == 4);
    REQUIRE(V->dofmap()->bs() == 1);

    // dof coordinates: the 4 corners of the unit square, one dof each,
    // laid out as (num_dofs, gdim) with gdim = 2.
    auto coords = V->tabulate_dof_coordinates(false);
    REQUIRE(coords.size() == 4 * 2);
    const double corners[4][2] = {{0, 0}, {1, 0}, {0, 1}, {1, 1}};
    for (int d = 0; d < 4; ++d) {
        REQUIRE(coords[2 * d + 0]
            == Catch::Approx(corners[d][0]).margin(1e-12));
        REQUIRE(coords[2 * d + 1]
            == Catch::Approx(corners[d][1]).margin(1e-12));
    }
}

TEST_CASE("FunctionSpace: tabulate_dof_coordinates transpose layout", "[fem]")
{
    auto V = p1_unit_square_space();
    // (gdim, num_dofs) layout: entry (component, dof).
    auto coords = V->tabulate_dof_coordinates(true);
    REQUIRE(coords.size() == 2 * 4);
    REQUIRE(coords[0 * 4 + 0] == Catch::Approx(0.0)); // x of dof 0
    REQUIRE(coords[0 * 4 + 1] == Catch::Approx(1.0)); // x of dof 1
    REQUIRE(coords[1 * 4 + 1] == Catch::Approx(0.0)); // y of dof 1
    REQUIRE(coords[1 * 4 + 3] == Catch::Approx(1.0)); // y of dof 3
}
