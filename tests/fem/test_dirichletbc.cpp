// hellofem::fem — DirichletBC unit tests
// SPDX-License-Identifier: MIT

#include "basis/element-families.h"
#include "basis/finite-element.h"
#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "common/IndexMap.h"
#include "fem/CoordinateElement.h"
#include "fem/DirichletBC.h"
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
#include <span>
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

    /// Boundary edges of the unit square: edges attached to exactly one
    /// cell (the interior diagonal is attached to two).
    std::vector<std::int32_t>
    boundary_edges(const fem::FunctionSpace<double>& V)
    {
        auto topology = V.mesh()->topology();
        auto e_to_c = topology->connectivity(1, 2);
        std::vector<std::int32_t> edges;
        for (std::int32_t e = 0; e < e_to_c->num_nodes(); ++e)
            if (e_to_c->num_links(e) == 1)
                edges.push_back(e);
        return edges;
    }

    /// Build a P1 FunctionSpace on a unit square with boundary facets
    /// created and connected.
    std::shared_ptr<fem::FunctionSpace<double>> p1_space()
    {
        auto mesh = mesh::create_unit_square(1);
        mesh->topology_mutable()->create_entities(1);
        mesh->topology_mutable()->create_connectivity(2, 1);
        mesh->topology_mutable()->create_connectivity(1, 2);
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

TEST_CASE("DirichletBC: locate dofs on boundary facets", "[fem]")
{
    auto V = p1_space();
    auto boundary = boundary_edges(*V);

    REQUIRE(boundary.size() == 4);
    auto dofs = fem::DirichletBC<double>::locate_dofs_topological(
        *V->mesh()->topology(), *V->dofmap(), 1, boundary);
    // All 4 P1 dofs (the corners) lie on the boundary.
    REQUIRE(dofs.size() == 4);
}

TEST_CASE("DirichletBC: constant value applied to a vector", "[fem]")
{
    auto V = p1_space();
    fem::Function<double> u(V);

    auto boundary = boundary_edges(*V);
    auto dofs = fem::DirichletBC<double>::locate_dofs_topological(
        *V->mesh()->topology(), *V->dofmap(), 1, boundary);
    REQUIRE(dofs.size() == 4);

    fem::DirichletBC<double> bc(2.0, dofs, V);

    // Apply to the coefficient vector: x[dof] = g.
    std::vector<double> x = u.x()->array();
    bc.set(std::span(x), std::nullopt, 1.0);
    for (std::int32_t d : dofs)
        REQUIRE(x[static_cast<std::size_t>(d)] == Approx(2.0));

    // mark_dofs flags the boundary dofs.
    std::vector<std::int8_t> markers(u.x()->array().size(), 0);
    bc.mark_dofs(std::span(markers));
    for (std::int32_t d : dofs)
        REQUIRE(markers[static_cast<std::size_t>(d)] == 1);
}

TEST_CASE("DirichletBC: alpha scaling in set", "[fem]")
{
    auto V = p1_space();
    auto boundary = boundary_edges(*V);
    auto dofs = fem::DirichletBC<double>::locate_dofs_topological(
        *V->mesh()->topology(), *V->dofmap(), 1, boundary);

    fem::DirichletBC<double> bc(4.0, dofs, V);
    std::vector<double> x(V->dofmap()->index_map->size_local(), 1.0);
    std::vector<double> x0(V->dofmap()->index_map->size_local(), 1.5);
    // x[dof] = alpha * (g - x0[dof]) with alpha = 0.5 -> 0.5 * (4 - 1.5).
    bc.set(std::span(x), std::span(x0), 0.5);
    for (std::int32_t d : dofs)
        REQUIRE(x[static_cast<std::size_t>(d)] == Approx(1.25));
}
