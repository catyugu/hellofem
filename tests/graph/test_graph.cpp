// hellofem::graph — unit tests
// SPDX-License-Identifier: MIT

#include <catch2/catch_test_macros.hpp>

#include "graph/AdjacencyList.h"
#include "graph/ordering.h"
#include "graph/partition.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <span>
#include <vector>

using namespace hellofem;
using namespace hellofem::graph;

TEST_CASE("AdjacencyList construction and access")
{
    std::vector<std::int32_t> data {1, 2, 0, 3, 2};
    std::vector<std::int32_t> offsets {0, 2, 4, 5};
    AdjacencyList<std::int32_t> list(data, offsets);
    REQUIRE(list.num_nodes() == 3);
    REQUIRE(list.num_links(0) == 2);
    REQUIRE(list.num_links(2) == 1);
    REQUIRE(list.links(0)[0] == 1);
    REQUIRE(list.links(1)[1] == 3);
    REQUIRE(list.array() == data);
    REQUIRE(list.offsets() == offsets);

    // Trivial self-loop list
    AdjacencyList<std::int32_t> self(4);
    REQUIRE(self.num_nodes() == 4);
    REQUIRE(self.links(3)[0] == 3);
}

TEST_CASE("regular_adjacency_list")
{
    std::vector<std::int32_t> data {0, 1, 1, 2, 2, 0};
    auto list = regular_adjacency_list(data, 2);
    REQUIRE(list.num_nodes() == 3);
    REQUIRE(list.num_links(0) == 2);
    REQUIRE(list.links(2)[1] == 0);
}

TEST_CASE("reorder_rcm reduces bandwidth on a path graph")
{
    // Path graph 0-1-2-...-9. RCM of a path is the same path (possibly
    // reversed), so the ordering must map the linear graph to a contiguous
    // sequence — every node's neighbours stay adjacent in the new numbering.
    const int n = 10;
    std::vector<std::int32_t> data, offsets;
    offsets.push_back(0);
    for (int i = 0; i < n; ++i) {
        if (i > 0)
            data.push_back(i - 1);
        if (i < n - 1)
            data.push_back(i + 1);
        offsets.push_back(offsets.back() + (i > 0) + (i < n - 1));
    }
    AdjacencyList<std::int32_t> graph(data, offsets);

    std::vector<std::int32_t> perm = reorder_rcm(graph);
    REQUIRE(perm.size() == n);
    // perm is a permutation of [0, n)
    std::vector<std::int32_t> sorted = perm;
    std::sort(sorted.begin(), sorted.end());
    REQUIRE(sorted == std::vector<std::int32_t> {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
}

TEST_CASE("partition_graph is single-process trivial")
{
    std::vector<std::int64_t> data {1, 0, 2, 0, 3, 1};
    std::vector<std::int32_t> offsets {0, 2, 4, 6};
    AdjacencyList<std::int64_t> graph(data, offsets);
    auto part = partition_graph(1, graph, false);
    REQUIRE(part.num_nodes() == 3);
    for (std::int32_t i = 0; i < 3; ++i)
        REQUIRE(part.links(i)[0] == 0);
}

TEST_CASE("build::compute_local_to_global and local_to_local")
{
    std::vector<std::int64_t> g {10, 20, 30};
    std::vector<std::int32_t> l {0, 1, 2};
    REQUIRE(build::compute_local_to_global(g, l)
        == std::vector<std::int64_t>({10, 20, 30}));

    // local0 indices map to global {10, 20, 30}; local1 the same global set
    // in a permuted order.
    std::vector<std::int64_t> l1 {30, 10, 20};
    auto l0l1 = build::compute_local_to_local(g, l1);
    REQUIRE(l0l1 == std::vector<std::int32_t>({1, 2, 0}));

    // build::distribute is a passthrough: 3 cells, 1 node each
    auto [cells, src, orig, ghosts] = build::distribute(
        std::span<const std::int64_t>(g), std::array<std::size_t, 2> {3, 1},
        AdjacencyList<std::int32_t>(1));
    REQUIRE(cells == std::vector<std::int64_t>({10, 20, 30}));
    REQUIRE(orig == std::vector<std::int64_t>({0, 1, 2}));
    REQUIRE(src == std::vector<int>({0, 0, 0}));
    REQUIRE(ghosts.empty());
}
