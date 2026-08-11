// hellofem::graph — single-process graph partition and build helpers
// SPDX-License-Identifier: MIT

#pragma once

#include "AdjacencyList.h"

#include <array>
#include <cstdint>
#include <functional>
#include <span>
#include <tuple>
#include <vector>

namespace hellofem::common {
    class IndexMap;
}

namespace hellofem::graph {

    /// Destination rank for each input node. Signature kept for parity with
    /// the mesh partitioner plumbing; a single-process library always assigns
    /// every node to rank 0.
    using partition_fn = std::function<AdjacencyList<std::int32_t>(
        int, int, const AdjacencyList<std::int64_t>&, bool)>;

    /// Partition a graph into `nparts` parts. Single-process: every node goes
    /// to partition 0, so the output is a constant-degree list of rank 0s.
    AdjacencyList<std::int32_t>
    partition_graph(int nparts, const AdjacencyList<std::int64_t>& local_graph,
                    bool ghosting);

    /// Single-process graph construction helpers. Every routine degenerates
    /// to an identity or local renumbering since there is no inter-rank
    /// communication.
    namespace build {
        /// Distribute an adjacency list to destination ranks. Single-process:
        /// returns the input unchanged (all nodes owned by rank 0).
        std::tuple<AdjacencyList<std::int64_t>, std::vector<int>,
                   std::vector<std::int64_t>, std::vector<int>>
        distribute(const AdjacencyList<std::int64_t>& list,
                   const AdjacencyList<std::int32_t>& destinations);

        /// Distribute a fixed-size (num_nodes x degree) node array.
        /// Single-process: returns the input unchanged.
        std::tuple<std::vector<std::int64_t>, std::vector<int>,
                   std::vector<std::int64_t>, std::vector<int>>
        distribute(std::span<const std::int64_t> list,
                   std::array<std::size_t, 2> shape,
                   const AdjacencyList<std::int32_t>& destinations);

        /// New global indices for ghost indices. Single-process: ghosts do
        /// not exist, so the result is empty.
        std::vector<std::int64_t>
        compute_ghost_indices(std::span<const std::int64_t> owned_indices,
                              std::span<const std::int64_t> ghost_indices,
                              std::span<const int> ghost_owners,
                              int num_threads);

        /// Map from local link index to global link index, given parallel
        /// arrays of matching global/local link entries.
        std::vector<std::int64_t>
        compute_local_to_global(std::span<const std::int64_t> global,
                                std::span<const std::int32_t> local);

        /// Map from local0 indices to local1 indices, given their two
        /// local-to-global maps (which must cover the same global set).
        std::vector<std::int32_t>
        compute_local_to_local(std::span<const std::int64_t> local0_to_global,
                               std::span<const std::int64_t> local1_to_global);
    } // namespace build

} // namespace hellofem::graph
