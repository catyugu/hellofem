// hellofem::mesh — build the mesh dual graph (cell-cell via facets)
// SPDX-License-Identifier: MIT

#include "graphbuild.h"

#include "cell_types.h"
#include "common/Timer.h"
#include "common/sort.h"
#include "spdlog/spdlog.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "oneapi/tbb/blocked_range.h"
#include "oneapi/tbb/parallel_for.h"

using namespace hellofem;
using namespace hellofem::mesh;

//-----------------------------------------------------------------------------
std::tuple<graph::AdjacencyList<std::int32_t>,
    std::vector<std::int64_t>, int, std::vector<std::int32_t>>
mesh::build_local_dual_graph(
    std::span<const CellType> celltypes,
    const std::vector<std::span<const std::int64_t>>& cells,
    std::optional<std::int32_t> max_facet_to_cell_links, int num_threads)
{
    if (num_threads < 1)
        throw std::runtime_error("num_threads must be >= 1.");

    spdlog::info("Build local part of mesh dual graph");
    common::Timer timer("Compute local part of mesh dual graph");

    const std::size_t ncells_local = std::accumulate(
        cells.begin(), cells.end(), std::size_t(0),
        [](std::size_t s, std::span<const std::int64_t> c) {
            return s + c.size();
        });
    if (ncells_local == 0)
        return {graph::AdjacencyList<std::int32_t>(0),
            std::vector<std::int64_t>(), 0, std::vector<std::int32_t>()};

    if (cells.size() != celltypes.size())
        throw std::runtime_error(
            "Number of cell types must match number of cell arrays.");

    const int tdim = cell_dim(celltypes.front());

    // Cell indexing offsets per cell type and the maximum number of
    // vertices per facet.
    std::vector<std::int32_t> cell_offsets {0};
    cell_offsets.reserve(cells.size() + 1);
    int max_vertices_per_facet = 0;
    std::size_t facet_count = 0;
    for (std::size_t j = 0; j < cells.size(); ++j) {
        const CellType cell_type = celltypes[j];
        assert(tdim == cell_dim(cell_type));
        const int num_cell_vertices = cell_num_entities(cell_type, 0);
        const int num_cell_facets = cell_num_entities(cell_type, tdim - 1);

        const std::int32_t num_cells
            = cells[j].size() / num_cell_vertices;
        cell_offsets.push_back(cell_offsets.back() + num_cells);
        facet_count += num_cell_facets * num_cells;

        const graph::AdjacencyList<std::int32_t> cell_facets
            = get_entity_vertices(cell_type, tdim - 1);
        for (int node = 0; node < cell_facets.num_nodes(); ++node)
            max_vertices_per_facet
                = std::max(max_vertices_per_facet, cell_facets.num_links(node));
    }

    // Build a flattened list of all facets, defined by sorted vertices,
    // with the attached cell index. Stored column-major so the sort
    // needs no per-column extraction copy.
    //
    //   facets[0..max_vertices_per_facet-1] : vertex columns
    //   facets[max_vertices_per_facet]      : attached cell index
    auto build_facets_fn
        = [](int num_vertices_per_facet_max, int num_cell_vertices,
              std::size_t cell_offset, std::size_t facet_offset,
              const graph::AdjacencyList<int>& cell_facets,
              std::span<const std::int64_t> cells,
              std::span<const std::span<std::int64_t>> facets) {
              constexpr std::int64_t padding_value = -1;
              std::vector<std::int64_t> row(num_vertices_per_facet_max);
              for (std::size_t c = 0; c < cells.size() / num_cell_vertices; ++c) {
                  auto v = cells.subspan(num_cell_vertices * c, num_cell_vertices);
                  for (int f = 0; f < cell_facets.num_nodes(); ++f) {
                      const std::span facet_vertices = cell_facets.links(f);
                      std::ranges::transform(
                          facet_vertices, row.begin(),
                          [v](auto idx) { return v[idx]; });

                      // Sort the facet's vertices. Hot loop: hand-unroll the
                      // common 2-, 3- and 4-vertex cases.
                      auto it = std::next(row.begin(), facet_vertices.size());
                      if (facet_vertices.size() == 2) {
                          if (row[0] > row[1])
                              std::swap(row[0], row[1]);
                      }
                      else if (facet_vertices.size() == 3) {
                          if (row[0] > row[1])
                              std::swap(row[0], row[1]);
                          if (row[1] > row[2])
                              std::swap(row[1], row[2]);
                          if (row[0] > row[1])
                              std::swap(row[0], row[1]);
                      }
                      else if (facet_vertices.size() == 4) {
                          if (row[0] > row[1])
                              std::swap(row[0], row[1]);
                          if (row[2] > row[3])
                              std::swap(row[2], row[3]);
                          if (row[0] > row[2])
                              std::swap(row[0], row[2]);
                          if (row[1] > row[3])
                              std::swap(row[1], row[3]);
                          if (row[1] > row[2])
                              std::swap(row[1], row[2]);
                      }
                      else
                          std::sort(row.begin(), it);
                      std::fill(it, row.end(), padding_value);

                      const std::size_t idx
                          = facet_offset + c * cell_facets.num_nodes() + f;
                      for (int k = 0; k < num_vertices_per_facet_max; ++k)
                          facets[k][idx] = row[k];
                      facets[num_vertices_per_facet_max][idx] = c + cell_offset;
                  }
              }
          };

    const int shape1 = max_vertices_per_facet + 1;
    std::vector<std::int64_t> facets_storage(facet_count * shape1);
    std::vector<std::span<std::int64_t>> facets(shape1);
    for (int col = 0; col < shape1; ++col) {
        facets[col] = std::span<std::int64_t>(
            facets_storage.data() + col * facet_count, facet_count);
    }

    std::size_t facet_offset = 0;
    for (std::size_t j = 0; j < cells.size(); ++j) {
        const CellType cell_type = celltypes[j];
        const int num_cell_vertices = cell_num_entities(cell_type, 0);
        const graph::AdjacencyList<int> cell_facets
            = get_entity_vertices(cell_type, tdim - 1);
        const std::span _cells = cells[j];
        const std::size_t num_cells_j = _cells.size() / num_cell_vertices;

        tbb::parallel_for(
            tbb::blocked_range<std::size_t>(0, num_cells_j),
            [&, max_vertices_per_facet, num_cell_vertices,
                cell_facets](const tbb::blocked_range<std::size_t>& range) {
                const std::size_t c0 = range.begin();
                const std::size_t c1 = range.end();
                build_facets_fn(max_vertices_per_facet, num_cell_vertices,
                    cell_offsets[j] + c0,
                    facet_offset + c0 * cell_facets.num_nodes(), cell_facets,
                    _cells.subspan(
                        c0 * num_cell_vertices,
                        (c1 - c0) * num_cell_vertices),
                    std::span<const std::span<std::int64_t>>(facets));
            },
            tbb::simple_partitioner {});

        facet_offset += num_cells_j * cell_facets.num_nodes();
    }

    // Sort facets by vertex key (excluding the trailing cell column).
    std::vector<std::span<const std::int64_t>> sort_cols(
        facets.begin(), std::next(facets.begin(), max_vertices_per_facet));
    const std::vector<std::int32_t> perm
        = sort_by_perm(std::span<std::span<const std::int64_t>>(sort_cols));

    // Iterate over the sorted facets. Facets shared by more than one
    // cell lead to a graph edge; facets with fewer than
    // `max_facet_to_cell_links` cells are stored as candidates for
    // facets shared across processes (a no-op here: single process).
    std::vector<std::int64_t> unmatched_facets;
    std::vector<std::int32_t> local_cells;
    std::vector<std::array<std::int32_t, 2>> edges;
    {
        auto facet_vertex
            = [&facets](std::size_t idx, int col) { return facets[col][idx]; };
        auto facet_cell = [&facets, max_vertices_per_facet](std::size_t idx) {
            return static_cast<std::int32_t>(facets[max_vertices_per_facet][idx]);
        };
        auto facets_equal
            = [&facet_vertex, max_vertices_per_facet](std::size_t idx0,
                  std::size_t idx1) {
                  for (int col = 0; col < max_vertices_per_facet; ++col)
                      if (facet_vertex(idx0, col) != facet_vertex(idx1, col))
                          return false;
                  return true;
              };

        for (auto it = perm.begin(); it != perm.end();) {
            const std::size_t facet_index = *it;
            auto matching_facets = std::ranges::subrange(it,
                std::find_if_not(it, perm.end(),
                    [facet_index, &facets_equal](auto idx) {
                        return facets_equal(facet_index, idx);
                    }));

            const std::int32_t cell_count = matching_facets.size();
            assert(cell_count >= 1);
            if (!max_facet_to_cell_links or cell_count < *max_facet_to_cell_links) {
                for (std::int32_t i = 0; i < cell_count; ++i) {
                    const std::size_t idx = *std::next(it, i);
                    for (int col = 0; col < max_vertices_per_facet; ++col)
                        unmatched_facets.push_back(facet_vertex(idx, col));
                    local_cells.push_back(facet_cell(idx));
                }
            }

            // Add dual graph edges (one direction; the reverse is added
            // when building the adjacency list below).
            for (auto facet_a_it = it; facet_a_it != matching_facets.end();
                ++facet_a_it) {
                const std::int32_t cell_a = facet_cell(*facet_a_it);
                for (auto facet_b_it = std::next(facet_a_it);
                    facet_b_it != matching_facets.end(); ++facet_b_it) {
                    const std::int32_t cell_b = facet_cell(*facet_b_it);
                    edges.push_back({cell_a, cell_b});
                }
            }

            it = matching_facets.end();
        }
    }

    // Build the adjacency list with both directions of every edge.
    std::vector<std::int32_t> num_links(cell_offsets.back(), 0);
    for (auto [a, b] : edges) {
        ++num_links[a];
        ++num_links[b];
    }
    std::vector<std::int32_t> offsets(num_links.size() + 1, 0);
    std::partial_sum(num_links.cbegin(), num_links.cend(),
        std::next(offsets.begin()));
    std::vector<std::int32_t> data(offsets.back());
    std::ranges::for_each(edges,
        [&data, pos = offsets](auto e) mutable {
            data[pos[e[0]]++] = e[1];
            data[pos[e[1]]++] = e[0];
        });

    return {graph::AdjacencyList(std::move(data), std::move(offsets)),
        std::move(unmatched_facets), max_vertices_per_facet,
        std::move(local_cells)};
}
//-----------------------------------------------------------------------------
graph::AdjacencyList<std::int64_t>
mesh::build_dual_graph(std::span<const CellType> celltypes,
    const std::vector<std::span<const std::int64_t>>& cells,
    std::optional<std::int32_t> max_facet_to_cell_links, int num_threads)
{
    spdlog::info("Building mesh dual graph");

    // Single process: no non-local (cross-process) facet matching is
    // needed, so the dual graph is exactly the local part.
    auto [local_graph, facets, shape1, fcells] = build_local_dual_graph(
        celltypes, cells, max_facet_to_cell_links, num_threads);

    return graph::AdjacencyList(
        std::vector<std::int64_t>(local_graph.array().begin(),
            local_graph.array().end()),
        local_graph.offsets());
}
//-----------------------------------------------------------------------------
