// hellofem::fem — DofMap unit tests
// SPDX-License-Identifier: MIT

#include "fem/CoordinateElement.h"
#include "fem/DofMap.h"
#include "fem/FiniteElement.h"
#include "fem/dofmapbuilder.h"

#include "basis/finite-element.h"
#include "catch2/catch_test_macros.hpp"
#include "common/IndexMap.h"
#include "graph/AdjacencyList.h"
#include "mesh/Topology.h"
#include "mesh/cell_types.h"

#include <cstdint>
#include <memory>
#include <span>
#include <utility>
#include <vector>

using namespace hellofem;

namespace {

    using B = basis::element::family;
    using C = basis::cell::type;
    using LV = basis::element::lagrange_variant;
    using DV = basis::element::dpc_variant;

    /// Build a 2-triangle topology: cells {0,1,2} and {0,2,3}, sharing
    /// vertex 2, with edge (dim 1) entities and connectivities created.
    std::shared_ptr<mesh::Topology> triangle_topology()
    {
        const std::vector<std::int64_t> cells = {0, 1, 2, 0, 2, 3};
        const std::vector<std::int64_t> orig = {0, 1};
        auto topology = std::make_shared<mesh::Topology>(
            mesh::create_topology(std::span<const std::int64_t>(cells),
                std::span<const std::int64_t>(orig),
                mesh::CellType::triangle, 1));

        // Edges + cell<->edge connectivities (needed by some dof layouts).
        topology->create_entities(1);
        topology->create_connectivity(2, 1);
        topology->create_connectivity(1, 2);
        return topology;
    }

    /// Scalar P1 dof layout on a triangle.
    fem::ElementDofLayout p1_layout()
    {
        return fem::CoordinateElement<double>(
            mesh::CellType::triangle, 1, LV::equispaced)
            .create_dof_layout();
    }

    /// Blocked (vector, 3 components) P1 dof layout on a triangle.
    fem::ElementDofLayout p1_vector_layout()
    {
        basis::FiniteElement<double> base = basis::create_element<double>(
            B::P, C::triangle, 1, LV::equispaced, DV::unset, false);
        fem::FiniteElement<double> vfe(base, std::vector<std::size_t> {3});
        return vfe.create_dof_layout();
    }

    /// Build a DofMap on the 2-triangle topology for a scalar P1 element.
    fem::DofMap build_p1_dofmap(const mesh::Topology& topology)
    {
        auto [imap, bs, dofmaps]
            = fem::build_dofmap_data(topology, {p1_layout()}, nullptr);
        return fem::DofMap(p1_layout(),
            std::make_shared<common::IndexMap>(std::move(imap)), bs,
            std::move(dofmaps.front()), bs);
    }

} // namespace

TEST_CASE("DofMap: scalar P1 on two triangles", "[fem]")
{
    auto topology = triangle_topology();
    fem::DofMap dmap = build_p1_dofmap(*topology);

    // 4 vertices -> 4 dofs
    REQUIRE(dmap.index_map->size_local() == 4);
    REQUIRE(dmap.bs() == 1);
    REQUIRE(dmap.index_map_bs() == 1);

    // Two cells, 3 dofs each
    auto dofs0 = dmap.cell_dofs(0);
    auto dofs1 = dmap.cell_dofs(1);
    REQUIRE(dofs0.size() == 3);
    REQUIRE(dofs1.size() == 3);

    // Vertex 2 is shared between both cells: its dof index appears twice.
    REQUIRE(dofs0[2] == dofs1[1]);
    for (std::int32_t d : dofs0)
        REQUIRE(d >= 0);
    for (std::int32_t d : dofs1)
        REQUIRE(d >= 0);
    for (std::int32_t d : dofs0)
        REQUIRE(d < 4);
    for (std::int32_t d : dofs1)
        REQUIRE(d < 4);

    // map() is (num_cells, dofs_per_cell) = (2, 3)
    auto map = dmap.map();
    REQUIRE(map.extent(0) == 2);
    REQUIRE(map.extent(1) == 3);
}

TEST_CASE("DofMap: extract_sub_dofmap on a blocked (vector) element", "[fem]")
{
    auto topology = triangle_topology();

    auto [imap, bs, dofmaps]
        = fem::build_dofmap_data(*topology, {p1_vector_layout()}, nullptr);
    fem::DofMap dmap(p1_vector_layout(),
        std::make_shared<common::IndexMap>(std::move(imap)), bs,
        std::move(dofmaps.front()), bs);

    REQUIRE(dmap.bs() == 3);
    REQUIRE(dmap.index_map_bs() == 3);
    REQUIRE(dmap.map().extent(1) == 3); // scalar dofs per cell

    // Extract component 0: a scalar P1 sub-dofmap. Its dofs are the
    // physical (blocked) dofs of the first component.
    std::vector<int> component = {0};
    fem::DofMap sub = dmap.extract_sub_dofmap(component);
    REQUIRE(sub.bs() == 1);
    REQUIRE(sub.index_map_bs() == 3); // still references the parent map

    auto sub0 = sub.cell_dofs(0);
    auto sub1 = sub.cell_dofs(1);
    REQUIRE(sub0.size() == 3);
    REQUIRE(sub1.size() == 3);
    // Physical dof = 3 * scalar dof for component 0.
    REQUIRE(sub0[0] == 3 * dmap.cell_dofs(0)[0]);
    REQUIRE(sub0[1] == 3 * dmap.cell_dofs(0)[1]);
    REQUIRE(sub1[1] == 3 * dmap.cell_dofs(1)[1]);
}

TEST_CASE("DofMap: collapse of a scalar dofmap is the identity", "[fem]")
{
    auto topology = triangle_topology();
    fem::DofMap dmap = build_p1_dofmap(*topology);

    auto [collapsed, collapsed_map] = dmap.collapse(*topology);

    // The P1 dofmap is already collapsed (dofs 0..3 are contiguous and
    // owned).
    REQUIRE(collapsed.index_map->size_local() == 4);
    REQUIRE(collapsed.bs() == 1);
    REQUIRE(collapsed.index_map_bs() == 1);

    // Collapsed dof i maps back to original dof i.
    REQUIRE(collapsed_map == std::vector<std::int32_t> {0, 1, 2, 3});

    // Cell dofs are unchanged.
    auto c0 = collapsed.cell_dofs(0);
    REQUIRE(std::vector(c0.begin(), c0.end())
        == std::vector(dmap.cell_dofs(0).begin(),
            dmap.cell_dofs(0).end()));
}

TEST_CASE("DofMap: transpose_dofmap collects flat dof positions", "[fem]")
{
    // dofmap: cell 0 -> {0,1}, cell 1 -> {1,2}, cell 2 -> {2,3}
    std::vector<std::int32_t> data = {0, 1, 1, 2, 2, 3};
    md::mdspan<const std::int32_t, md::dextents<std::size_t, 2>> dofmap(
        data.data(), 3, 2);

    auto adj = fem::transpose_dofmap(dofmap, 3);

    // For each dof, the flat positions (cell*2 + dof) where it appears:
    // dof 0 -> {0}, dof 1 -> {1,2}, dof 2 -> {3,4}, dof 3 -> {5}
    REQUIRE(adj.num_nodes() == 4);
    REQUIRE(std::vector(adj.links(0).begin(), adj.links(0).end())
        == std::vector<std::int32_t> {0});
    REQUIRE(std::vector(adj.links(1).begin(), adj.links(1).end())
        == std::vector<std::int32_t> {1, 2});
    REQUIRE(std::vector(adj.links(2).begin(), adj.links(2).end())
        == std::vector<std::int32_t> {3, 4});
    REQUIRE(std::vector(adj.links(3).begin(), adj.links(3).end())
        == std::vector<std::int32_t> {5});
}

TEST_CASE("DofMap: operator== compares data and block sizes", "[fem]")
{
    auto topology = triangle_topology();
    fem::DofMap a = build_p1_dofmap(*topology);
    fem::DofMap b = build_p1_dofmap(*topology);
    REQUIRE(a == b);
}
