// hellofem::fem — sparsitybuild unit tests
// SPDX-License-Identifier: MIT

#include "fem/DofMap.h"
#include "fem/CoordinateElement.h"
#include "fem/dofmapbuilder.h"
#include "fem/sparsitybuild.h"

#include "catch2/catch_test_macros.hpp"
#include "common/IndexMap.h"
#include "la/SparsityPattern.h"
#include "mesh/Topology.h"
#include "mesh/cell_types.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <utility>
#include <vector>

using namespace hellofem;

namespace {

    using LV = basis::element::lagrange_variant;

    /// Build a 2-triangle topology sharing vertex 2, with edges created.
    std::shared_ptr<mesh::Topology> triangle_topology()
    {
        const std::vector<std::int64_t> cells = {0, 1, 2, 0, 2, 3};
        const std::vector<std::int64_t> orig = {0, 1};
        auto topology = std::make_shared<mesh::Topology>(
            mesh::create_topology(std::span<const std::int64_t>(cells),
                std::span<const std::int64_t>(orig),
                mesh::CellType::triangle, 1));
        topology->create_entities(1);
        topology->create_connectivity(2, 1);
        return topology;
    }

    /// Scalar P1 dof layout on a triangle.
    fem::ElementDofLayout p1_layout()
    {
        return fem::CoordinateElement<double>(
            mesh::CellType::triangle, 1, LV::equispaced)
            .create_dof_layout();
    }

    /// Build a P1 DofMap on the given topology.
    fem::DofMap build_p1_dofmap(const mesh::Topology& topology)
    {
        auto [imap, bs, dofmaps]
            = fem::build_dofmap_data(topology, {p1_layout()}, nullptr);
        return fem::DofMap(p1_layout(),
            std::make_shared<common::IndexMap>(std::move(imap)), bs,
            std::move(dofmaps.front()), bs);
    }

} // namespace

TEST_CASE("sparsitybuild::cells builds the dof coupling graph", "[fem]")
{
    auto topology = triangle_topology();
    fem::DofMap dmap = build_p1_dofmap(*topology);

    auto imap = dmap.index_map;
    la::SparsityPattern pattern(imap);

    // Both cells, one cell-pair per entry.
    std::vector<int> cells = {0, 1};
    fem::sparsitybuild::cells(pattern, std::pair(cells, cells),
        {std::cref(dmap), std::cref(dmap)});

    pattern.finalize();

    // Each cell couples all its vertex dofs: cell0 {v0,v1,v2} x itself,
    // cell1 {v0,v2,v3} x itself. Deduplicated rows:
    //   v0: {v0,v1,v2,v3}, v1: {v0,v1,v2}, v2: {v0,v1,v2,v3}, v3: {v0,v2,v3}
    // Total nonzeros = 4 + 3 + 4 + 3 = 14.
    REQUIRE(pattern.num_nonzeros() == 14);

    const auto [cols, offsets] = pattern.graph();
    REQUIRE(offsets.size() == 5); // 4 rows + 1
    REQUIRE(offsets[0] == 0);
    REQUIRE(offsets[4] == 14);

    // Each row's columns are sorted.
    for (std::size_t i = 0; i < 4; ++i)
        REQUIRE(std::is_sorted(cols.begin() + offsets[i],
            cols.begin() + offsets[i + 1]));
}

TEST_CASE("sparsitybuild::interior_facets couples the two cells", "[fem]")
{
    auto topology = triangle_topology();
    fem::DofMap dmap = build_p1_dofmap(*topology);

    auto imap = dmap.index_map;
    la::SparsityPattern pattern(imap);

    // One interior facet shared between cell 0 and cell 1.
    std::vector<std::int32_t> cells0 = {0, 1};
    std::vector<std::int32_t> cells1 = {1, 0};
    fem::sparsitybuild::interior_facets(pattern,
        {std::span(cells0), std::span(cells1)},
        {std::cref(dmap), std::cref(dmap)});

    pattern.finalize();

    // The four (test, trial) blocks fully couple both cells' dofs, so the
    // graph is complete over all 4 vertices: 16 nonzeros.
    REQUIRE(pattern.num_nonzeros() == 16);

    // Cross-cell coupling: row of a dof exclusive to cell 0 (v1) must
    // reach a dof exclusive to cell 1 (v3).
    const auto [cols, offsets] = pattern.graph();
    const std::int32_t v1 = dmap.cell_dofs(0)[1];
    const std::int32_t v3 = dmap.cell_dofs(1)[2];
    REQUIRE(v1 != v3);
    const auto row = cols.subspan(offsets[v1], offsets[v1 + 1] - offsets[v1]);
    REQUIRE(std::find(row.begin(), row.end(), v3) != row.end());
}
