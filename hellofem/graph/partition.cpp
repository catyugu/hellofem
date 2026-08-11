// hellofem::graph — single-process graph partition and build helpers
// SPDX-License-Identifier: MIT

#include "partition.h"

#include "../common/Timer.h"
#include "../common/log.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

using namespace hellofem;
using namespace hellofem::graph;

AdjacencyList<std::int32_t>
graph::partition_graph(int nparts, const AdjacencyList<std::int64_t>& local_graph,
                       bool)
{
    common::Timer t("Partition graph (single-process)");
    spdlog::debug("Partitioning graph across {} partitions", nparts);

    // Every node is owned by the single process, so it lands in partition 0.
    const std::int32_t n = local_graph.num_nodes();
    std::vector<std::int32_t> dest(n, 0);
    std::vector<std::int32_t> offsets(n + 1);
    std::iota(offsets.begin(), offsets.end(), 0);
    return AdjacencyList(std::move(dest), std::move(offsets));
}

std::tuple<AdjacencyList<std::int64_t>, std::vector<int>,
           std::vector<std::int64_t>, std::vector<int>>
graph::build::distribute(const AdjacencyList<std::int64_t>& list,
                         const AdjacencyList<std::int32_t>&)
{
    // No redistribution: everything already lives on this (only) process.
    std::vector<int> src(list.num_nodes(), 0);
    std::vector<std::int64_t> original(list.num_nodes());
    std::iota(original.begin(), original.end(), 0);
    return {list, std::move(src), std::move(original), {}};
}

std::tuple<std::vector<std::int64_t>, std::vector<int>,
           std::vector<std::int64_t>, std::vector<int>>
graph::build::distribute(std::span<const std::int64_t> list,
                         std::array<std::size_t, 2> shape,
                         const AdjacencyList<std::int32_t>&)
{
    std::vector<std::int64_t> cells(list.begin(), list.end());
    std::vector<int> src(shape[0], 0);
    std::vector<std::int64_t> original(shape[0]);
    std::iota(original.begin(), original.end(), 0);
    return {std::move(cells), std::move(src), std::move(original), {}};
}

std::vector<std::int64_t>
graph::build::compute_ghost_indices(std::span<const std::int64_t>,
                                    std::span<const std::int64_t>,
                                    std::span<const int>, int)
{
    // No ghost indices in a single-process world.
    return {};
}

std::vector<std::int64_t>
graph::build::compute_local_to_global(std::span<const std::int64_t> global,
                                      std::span<const std::int32_t> local)
{
    common::Timer t("Compute local-to-global links");

    if (global.empty() and local.empty())
        return {};
    if (global.size() != local.size())
        throw std::runtime_error("Data size mismatch.");

    std::int32_t max_local_idx = *std::ranges::max_element(local);
    std::vector<std::int64_t> local_to_global_list(max_local_idx + 1, -1);
    for (std::size_t i = 0; i < local.size(); ++i) {
        if (local_to_global_list[local[i]] == -1)
            local_to_global_list[local[i]] = global[i];
    }
    return local_to_global_list;
}

std::vector<std::int32_t>
graph::build::compute_local_to_local(std::span<const std::int64_t> local0_to_global,
                                     std::span<const std::int64_t> local1_to_global)
{
    common::Timer t("Compute local-to-local map");
    assert(local0_to_global.size() == local1_to_global.size());

    // Inverse map: global -> local1
    std::vector<std::pair<std::int64_t, std::int32_t>> global_to_local1;
    global_to_local1.reserve(local1_to_global.size());
    for (std::size_t i = 0; i < local1_to_global.size(); ++i)
        global_to_local1.push_back({local1_to_global[i], (std::int32_t)i});
    std::ranges::sort(global_to_local1);

    std::vector<std::int32_t> local0_to_local1;
    local0_to_local1.reserve(local0_to_global.size());
    std::ranges::transform(local0_to_global,
                           std::back_inserter(local0_to_local1),
                           [&global_to_local1](auto g) {
                               auto it = std::ranges::lower_bound(
                                   global_to_local1, g, std::ranges::less(),
                                   [](auto e) { return e.first; });
                               assert(it != global_to_local1.end()
                                      and it->first == g);
                               return it->second;
                           });
    return local0_to_local1;
}
