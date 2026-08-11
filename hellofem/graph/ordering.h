// hellofem::graph — Reverse Cuthill-McKee ordering
// SPDX-License-Identifier: MIT

#pragma once

#include "AdjacencyList.h"

#include <cstdint>
#include <vector>

namespace hellofem::graph {

    /// Reorder a graph with the Reverse Cuthill-McKee (RCM) algorithm.
    ///
    /// The pseudo-peripheral root is found with the George-Liu "double
    /// sweep" heuristic, a level structure is built, each level is numbered
    /// in increasing degree order, and the numbering is reversed. Runs in
    /// O(V + E).
    ///
    /// @param[in] graph The graph to reorder.
    /// @return Reordering `map`, where `map[i]` is the new index of node `i`.
    std::vector<std::int32_t>
    reorder_rcm(const AdjacencyList<std::int32_t>& graph);

} // namespace hellofem::graph
