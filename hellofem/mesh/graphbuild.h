// hellofem::mesh — build the mesh dual graph (cell-cell via facets)
// SPDX-License-Identifier: MIT

#pragma once

#include "cell_types.h"
#include "graph/AdjacencyList.h"

#include <cstdint>
#include <optional>
#include <span>
#include <tuple>
#include <vector>

namespace hellofem::mesh {

    /// Compute the local part of the dual graph (cell-cell connections
    /// via facets) and the facets shared by at most
    /// `max_facet_to_cell_links` cells (candidates for facets on the
    /// domain boundary or shared with another process).
    ///
    /// @param[in] celltypes Cell types.
    /// @param[in] cells Cell vertex lists (flattened, one per cell
    /// type, using global vertex indices).
    /// @param[in] max_facet_to_cell_links Bound on the number of cells a
    /// facet must be connected to in order to be considered *matched*,
    /// i.e. not connected to any cells elsewhere. `std::nullopt`
    /// (no bound) treats every facet as unmatched. Defaults to 2 for
    /// manifold meshes.
    /// @param[in] num_threads Number of threads to use (>= 1).
    /// @return (dual graph, unmatched facets with sorted vertices
    /// flattened row-major, number of columns of the facet data, cells
    /// attached to each returned facet).
    std::tuple<graph::AdjacencyList<std::int32_t>,
        std::vector<std::int64_t>, int, std::vector<std::int32_t>>
    build_local_dual_graph(std::span<const CellType> celltypes,
        const std::vector<std::span<const std::int64_t>>& cells,
        std::optional<std::int32_t> max_facet_to_cell_links, int num_threads);

    /// Build the mesh dual graph (cell-cell connections via facets).
    /// Single-process: no cross-process facet matching is needed.
    ///
    /// @param[in] celltypes Cell types.
    /// @param[in] cells Cell vertex lists (flattened, one per cell
    /// type, using global vertex indices).
    /// @param[in] max_facet_to_cell_links As in build_local_dual_graph.
    /// @param[in] num_threads Number of threads to use (>= 1).
    graph::AdjacencyList<std::int64_t>
    build_dual_graph(std::span<const CellType> celltypes,
        const std::vector<std::span<const std::int64_t>>& cells,
        std::optional<std::int32_t> max_facet_to_cell_links,
        int num_threads = 1);

} // namespace hellofem::mesh
