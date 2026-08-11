// hellofem::mesh — compute entities and connectivity of a topology
// SPDX-License-Identifier: MIT

#include "topologycomputation.h"

#include "Topology.h"
#include "cell_types.h"
#include "common/Timer.h"
#include "common/sort.h"
#include "spdlog/spdlog.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "oneapi/tbb/blocked_range.h"
#include "oneapi/tbb/parallel_for.h"

using namespace hellofem;
using namespace hellofem::mesh;

namespace {

    /// Hash for a small fixed-size array of integers (used as a map key
    /// for entity vertex sets). std::array has no standard hash.
    template <typename T, std::size_t N>
    struct ArrayHash {
        std::size_t operator()(const std::array<T, N>& a) const noexcept
        {
            std::size_t h = 0;
            for (auto v : a)
                h ^= std::hash<T> {}(v) + 0x9e3779b97f4a7c15ULL + (h << 6)
                    + (h >> 2);
            return h;
        }
    };

    /// Build the list of entity instances of a given type from cells.
    ///
    /// For each cell in the (contiguous) input range and each entity of
    /// type `entity_type` on it, the cell-local entity vertices are
    /// extracted and written to `entity_list` (rows of
    /// `num_vertices_per_entity`), oriented so that each entity's
    /// vertices are globally sorted (the lowest global vertex first;
    /// for quadrilaterals the vertex opposite the lowest is last). The
    /// globally-sorted keys are scattered into `entity_list_sorted`
    /// (column-major) for the deduplication pass. Thread-safe: each
    /// caller writes a disjoint slice.
    ///
    /// @param[out] entity_list Oriented entity-vertex rows.
    /// @param[in] entity_offset Index of the first entity this call
    /// writes into `entity_list_sorted`.
    /// @param[out] entity_list_sorted Sorted-key columns.
    /// @param[in] cells Cell-to-vertex connectivity (flattened).
    /// @param[in] num_cell_vertices Vertices per cell.
    /// @param[in] e_vertices Entity-to-vertices, where `links(e)[i]` is
    /// the local (to the cell) vertex index of the i-th vertex of
    /// entity `e`.
    /// @param[in] entity_type Type of the entities to extract.
    /// @param[in] cell_type_entities Local indices of entities of type
    /// `entity_type` in a cell.
    /// @param[in] vertex_index_map Index map of the vertices.
    void build_entity_list(std::span<std::int32_t> entity_list,
        std::size_t entity_offset,
        std::span<const std::span<std::int32_t>> entity_list_sorted,
        std::span<const std::int32_t> cells, std::size_t num_vertices_per_cell,
        const graph::AdjacencyList<std::int32_t>& e_vertices,
        CellType entity_type, const std::vector<std::int32_t>& cell_type_entities,
        const common::IndexMap& vertex_index_map)
    {
        const int num_vertices_per_entity = mesh::num_cell_vertices(entity_type);
        const int num_entities_per_cell = cell_type_entities.size();

        std::vector<std::int32_t> entity_vertices(num_vertices_per_entity);
        std::vector<std::int64_t> global_vertices(num_vertices_per_entity);
        std::vector<std::size_t> perm(num_vertices_per_entity);

        // Scratch for the sorted key row. 4 vertices is the largest entity
        // (quadrilateral) supported by any cell type.
        std::array<std::int32_t, 4> row_sorted_storage {};

        auto it_e = entity_list.begin();
        std::size_t entity_idx = entity_offset;
        const std::size_t num_cells = cells.size() / num_vertices_per_cell;
        for (std::size_t c = 0; c < num_cells; ++c) {
            auto vertices
                = cells.subspan(c * num_vertices_per_cell, num_vertices_per_cell);

            for (int e = 0; e < num_entities_per_cell; ++e) {
                auto ev = e_vertices.links(cell_type_entities[e]);
                assert(ev.size() == entity_vertices.size());
                for (std::size_t j = 0; j < ev.size(); ++j)
                    entity_vertices[j] = vertices[ev[j]];

                // Orient the entity: reorder vertices so that the lowest
                // global vertex is first.
                vertex_index_map.local_to_global(entity_vertices, global_vertices);

                auto elist = std::span(it_e, num_vertices_per_entity);
                auto elist_sorted
                    = std::span(row_sorted_storage.data(), num_vertices_per_entity);

                // Insertion sort over the local vertex indices, ordered by the
                // global vertex index. Edges and triangles (2 and 3 vertices)
                // are the overwhelmingly common cases; std::ranges::sort of a
                // 2- or 3-element array is disproportionately expensive here.
                std::iota(perm.begin(), perm.end(), 0);
                for (std::size_t i = 1; i < perm.size(); ++i) {
                    std::size_t j = i;
                    while (j > 0
                        and global_vertices[perm[j - 1]]
                            > global_vertices[perm[j]]) {
                        std::swap(perm[j - 1], perm[j]);
                        --j;
                    }
                }

                // For a quadrilateral, the vertex opposite the lowest
                // numbered vertex must be last.
                if (entity_type == CellType::quadrilateral) {
                    const std::size_t min_vertex_idx = perm[0];
                    const std::size_t opposite_vertex_index
                        = 3 - min_vertex_idx;
                    auto it = std::find(
                        perm.begin(), perm.end(), opposite_vertex_index);
                    assert(it != perm.end());
                    std::rotate(it, it + 1, perm.end());
                }

                for (std::size_t j = 0; j < ev.size(); ++j)
                    elist[j] = entity_vertices[perm[j]];

                // Sorted key (independent of orientation)
                std::copy(elist.begin(), elist.end(), elist_sorted.begin());
                std::ranges::sort(elist_sorted);
                for (int k = 0; k < num_vertices_per_entity; ++k)
                    entity_list_sorted[k][entity_idx] = elist_sorted[k];

                std::advance(it_e, num_vertices_per_entity);
                ++entity_idx;
            }
        }
    }

    /// Create an adjacency list from a sorted array of (node, link)
    /// pairs: `data[i]` stores node `data[i].first` with link
    /// `data[i].second`. Nodes must be grouped contiguously.
    template <typename U>
    graph::AdjacencyList<int> create_adj_list(U& data, std::int32_t size)
    {
        auto [unique_end, range_end] = std::ranges::unique(data);
        data.erase(unique_end, range_end);

        std::vector<int> array;
        array.reserve(data.size());
        std::ranges::transform(
            data, std::back_inserter(array), [](auto x) { return x.second; });

        std::vector<std::int32_t> offsets {0};
        offsets.reserve(size + 1);
        auto it = data.begin();
        for (std::int32_t e = 0; e < size; ++e) {
            auto it1 = std::find_if(it, data.end(),
                [e](auto x) { return x.first != e; });
            offsets.push_back(
                offsets.back() + std::ranges::distance(it, it1));
            it = it1;
        }

        return graph::AdjacencyList(std::move(array), std::move(offsets));
    }

    /// Compute the entities of dimension `dim` and type `entity_type`
    /// of a topology by key matching: build oriented entity-vertex
    /// lists for every (cell, local-entity) instance, deduplicate
    /// identical vertex keys, and number the unique entities.
    ///
    /// @param[in] cell_lists (CellType, cells) for each cell type.
    /// @param[in] vertex_index_map Index map of the vertices.
    /// @param[in] num_threads Number of threads to use.
    /// @return (cell-entity connectivity per cell type, entity-vertex
    /// connectivity, entity index map, inter-process entity indices).
    std::tuple<std::vector<std::shared_ptr<graph::AdjacencyList<std::int32_t>>>,
        graph::AdjacencyList<std::int32_t>, common::IndexMap,
        std::vector<std::int32_t>>
    compute_entities_by_key_matching(
        std::vector<std::pair<CellType, std::span<const std::int32_t>>>
            cell_lists,
        const common::IndexMap& vertex_index_map, CellType entity_type, int dim,
        [[maybe_unused]] int num_threads)
    {
        if (dim == 0)
            throw std::runtime_error(
                "Cannot create vertices for topology. Should already exist.");

        assert(cell_dim(entity_type) == dim);
        assert(num_threads > 0);

        common::Timer timer(
            std::format("Compute entities of dim = {}", dim));

        // Per cell type, the local entity indices that have type
        // `entity_type`, plus the instance offset within the global
        // entity list.
        std::vector<std::vector<std::int32_t>> cell_type_entities(
            cell_lists.size());
        std::vector<std::int32_t> cell_type_offsets {0};
        for (std::size_t k = 0; k < cell_lists.size(); ++k) {
            const CellType cell_type = cell_lists[k].first;
            for (int e = 0; e < cell_num_entities(cell_type, dim); ++e) {
                if (cell_entity_type(cell_type, dim, e) == entity_type)
                    cell_type_entities[k].push_back(e);
            }

            const std::size_t num_cells
                = cell_lists[k].second.size() / num_cell_vertices(cell_type);
            cell_type_offsets.push_back(cell_type_offsets.back()
                + num_cells * cell_type_entities[k].size());
        }

        const int num_vertices_per_entity = num_cell_vertices(entity_type);
        const std::int32_t num_instances = cell_type_offsets.back();
        std::vector<std::int32_t> entity_list(
            num_instances * num_vertices_per_entity);

        // Sorted-key columns (column-major, one span per vertex), used
        // only for deduplication below.
        std::vector<std::int32_t> entity_list_sorted_storage(
            num_instances * num_vertices_per_entity);
        std::vector<std::span<std::int32_t>> entity_list_sorted(
            num_vertices_per_entity);
        for (int col = 0; col < num_vertices_per_entity; ++col) {
            entity_list_sorted[col] = std::span<std::int32_t>(
                entity_list_sorted_storage.data() + col * num_instances,
                num_instances);
        }

        for (std::size_t k = 0; k < cell_lists.size(); ++k) {
            const CellType cell_type = cell_lists[k].first;
            const std::size_t num_vertices_per_cell = num_cell_vertices(cell_type);
            const auto e_vertices = get_entity_vertices(cell_type, dim);
            const int num_entities_per_cell = cell_type_entities[k].size();
            const std::size_t num_cells
                = cell_lists[k].second.size() / num_vertices_per_cell;

            tbb::parallel_for(
                tbb::blocked_range<std::size_t>(0, num_cells),
                [&, num_vertices_per_cell, num_entities_per_cell](
                    const tbb::blocked_range<std::size_t>& range) {
                    const std::size_t c0 = range.begin();
                    const std::size_t c1 = range.end();
                    const std::size_t offset = cell_type_offsets[k]
                            * num_vertices_per_entity
                        + c0 * num_vertices_per_entity * num_entities_per_cell;
                    const std::size_t count = (c1 - c0)
                        * num_vertices_per_entity * num_entities_per_cell;
                    const std::size_t entity_offset = cell_type_offsets[k]
                        + c0 * num_entities_per_cell;
                    build_entity_list(
                        std::span(entity_list.data() + offset, count),
                        entity_offset,
                        std::span<const std::span<std::int32_t>>(
                            entity_list_sorted),
                        cell_lists[k].second.subspan(
                            c0 * num_vertices_per_cell,
                            (c1 - c0) * num_vertices_per_cell),
                        num_vertices_per_cell, e_vertices, entity_type,
                        cell_type_entities[k], vertex_index_map);
                },
                tbb::simple_partitioner {});
        }

        // Sort the entity instances by their (globally sorted) vertex
        // key and label unique entities.
        std::vector<std::int32_t> entity_index(num_instances);
        std::int32_t entity_count = 0;
        {
            common::Timer t_number("Number entities");
            // Read-only view of the sorted-key columns for the sort.
            std::vector<std::span<const std::int32_t>> cols(
                entity_list_sorted.begin(), entity_list_sorted.end());
            const std::vector<std::int32_t> sort_order
                = sort_by_perm(std::span<std::span<const std::int32_t>>(cols));

            auto it = sort_order.begin();
            while (it != sort_order.end()) {
                const std::size_t idx0 = *it;
                auto it1 = std::find_if_not(
                    it, sort_order.end(),
                    [idx0, &entity_list_sorted,
                        num_vertices_per_entity](auto idx) -> bool {
                        for (int k = 0; k < num_vertices_per_entity; ++k)
                            if (entity_list_sorted[k][idx0]
                                != entity_list_sorted[k][idx])
                                return false;
                        return true;
                    });
                std::for_each(it, it1,
                    [&entity_index, entity_count](auto idx) {
                        entity_index[idx] = entity_count;
                    });
                it = it1;
                ++entity_count;
            }
        }

        // Single process: all entities are owned, so the local index map
        // is the identity numbering 0..entity_count-1 and there are no
        // inter-process entities.
        common::IndexMap index_map(0, entity_count);

        // Entity-vertex connectivity.
        std::vector<std::int32_t> ev_array(
            entity_count * num_vertices_per_entity);
        graph::AdjacencyList ev = graph::regular_adjacency_list(
            std::move(ev_array), num_vertices_per_entity);
        for (std::int32_t i = 0; i < num_instances; ++i) {
            std::copy_n(std::next(entity_list.begin(), i * num_vertices_per_entity),
                num_vertices_per_entity,
                ev.links(entity_index[i]).begin());
        }

        // Cell-entity connectivity per cell type.
        std::vector<std::shared_ptr<graph::AdjacencyList<std::int32_t>>> ce(
            cell_lists.size());
        for (std::size_t k = 0; k < cell_lists.size(); ++k) {
            if (!cell_type_entities[k].empty()) {
                // In the single-process build, entity_index holds the
                // final local indices, so the per-cell-type slice is
                // directly usable as the cell-entity list.
                auto begin = entity_index.begin() + cell_type_offsets[k];
                std::vector tmp(begin, begin + (cell_type_offsets[k + 1] - cell_type_offsets[k]));
                ce[k] = std::make_shared<graph::AdjacencyList<std::int32_t>>(
                    graph::regular_adjacency_list(
                        std::move(tmp), cell_type_entities[k].size()));
            }
        }

        return {std::move(ce), std::move(ev), std::move(index_map),
            std::vector<std::int32_t>()};
    }

    /// Compute the `d0 -> d1` connectivity from the transpose `d1 ->
    /// d0`.
    graph::AdjacencyList<std::int32_t>
    compute_from_transpose(const graph::AdjacencyList<std::int32_t>& c_d1_d0,
        const int num_entities_d0)
    {
        std::vector<std::int32_t> num_connections(num_entities_d0, 0);
        for (int e1 = 0; e1 < c_d1_d0.num_nodes(); ++e1)
            for (std::int32_t e0 : c_d1_d0.links(e1))
                num_connections[e0]++;

        std::vector<std::int32_t> offsets(num_connections.size() + 1, 0);
        std::partial_sum(num_connections.begin(), num_connections.end(),
            std::next(offsets.begin()));

        std::vector<std::int32_t> counter(num_connections.size(), 0);
        std::vector<std::int32_t> connections(offsets.back());
        for (int e1 = 0; e1 < c_d1_d0.num_nodes(); ++e1)
            for (std::int32_t e0 : c_d1_d0.links(e1))
                connections[offsets[e0] + counter[e0]++] = e1;

        return graph::AdjacencyList(std::move(connections), std::move(offsets));
    }

    /// Compute the `d0 -> d1` connectivity, where d0 > d1, by matching
    /// the vertices of each d0 entity against the vertices of the d1
    /// entities.
    graph::AdjacencyList<std::int32_t>
    compute_from_map(const graph::AdjacencyList<std::int32_t>& c_d0_0,
        const graph::AdjacencyList<std::int32_t>& c_d1_0)
    {
        // Map from sorted edge-vertex pair to edge index.
        std::unordered_map<std::array<std::int32_t, 2>, std::int32_t,
            ArrayHash<std::int32_t, 2>>
            edge_to_index;
        edge_to_index.reserve(c_d1_0.num_nodes());

        std::array<std::int32_t, 2> key;
        for (int e = 0; e < c_d1_0.num_nodes(); ++e) {
            std::span<const std::int32_t> v = c_d1_0.links(e);
            assert(v.size() == key.size());
            std::partial_sort_copy(v.begin(), v.end(), key.begin(), key.end());
            edge_to_index.insert({key, e});
        }

        std::vector<std::int32_t> connections;
        connections.reserve(c_d0_0.array().size());
        std::vector<std::int32_t> offsets(c_d0_0.offsets());

        // Reference edge-vertex tables for triangle and quadrilateral.
        const graph::AdjacencyList<int> tri_vertices_ref
            = get_entity_vertices(CellType::triangle, 1);
        const graph::AdjacencyList<int> quad_vertices_ref
            = get_entity_vertices(CellType::quadrilateral, 1);
        for (int e = 0; e < c_d0_0.num_nodes(); ++e) {
            auto e0 = c_d0_0.links(e);
            auto vref = (e0.size() == 3) ? &tri_vertices_ref : &quad_vertices_ref;
            for (std::size_t i = 0; i < e0.size(); ++i) {
                auto v = vref->links(i);
                for (int j = 0; j < 2; ++j)
                    key[j] = e0[v[j]];
                std::ranges::sort(key);
                auto it = edge_to_index.find(key);
                assert(it != edge_to_index.end());
                connections.push_back(it->second);
            }
        }

        connections.shrink_to_fit();
        return graph::AdjacencyList(std::move(connections), std::move(offsets));
    }

} // namespace

//-----------------------------------------------------------------------------
std::tuple<std::vector<std::shared_ptr<graph::AdjacencyList<std::int32_t>>>,
    std::shared_ptr<graph::AdjacencyList<std::int32_t>>,
    std::shared_ptr<common::IndexMap>, std::vector<std::int32_t>>
mesh::compute_entities(const Topology& topology, int dim, CellType entity_type,
    int num_threads)
{
    if (num_threads < 1)
        throw std::runtime_error("num_threads must be >= 1.");

    spdlog::info("Computing mesh entities of dimension {}", dim);

    // Vertices always exist.
    if (dim == 0)
        return {std::vector<std::shared_ptr<graph::AdjacencyList<std::int32_t>>>(),
            nullptr, nullptr, std::vector<std::int32_t>()};

    {
        auto idx = std::ranges::find(topology.entity_types(dim), entity_type);
        assert(idx != topology.entity_types(dim).end());
        const int index = std::ranges::distance(topology.entity_types(dim).begin(), idx);
        if (topology.connectivity({dim, index}, {0, 0}))
            return {std::vector<std::shared_ptr<graph::AdjacencyList<std::int32_t>>>(),
                nullptr, nullptr, std::vector<std::int32_t>()};
    }

    const int tdim = topology.dim();
    const std::vector<CellType> cell_types = topology.entity_types(tdim);
    std::vector<std::pair<CellType, std::span<const std::int32_t>>> cell_lists;
    cell_lists.reserve(cell_types.size());
    for (std::size_t i = 0; i < cell_types.size(); ++i) {
        auto cells = topology.connectivity({tdim, int(i)}, {0, 0});
        if (!cells)
            throw std::runtime_error("Cell connectivity missing.");
        cell_lists.push_back({cell_types[i], cells->array()});
    }

    auto vertex_map = topology.index_map(0);
    assert(vertex_map);

    auto [d0, d1, im, interprocess_entities]
        = compute_entities_by_key_matching(
            std::move(cell_lists), *vertex_map, entity_type, dim, num_threads);

    return {d0,
        std::make_shared<graph::AdjacencyList<std::int32_t>>(std::move(d1)),
        std::make_shared<common::IndexMap>(std::move(im)),
        std::move(interprocess_entities)};
}
//-----------------------------------------------------------------------------
std::array<std::shared_ptr<graph::AdjacencyList<std::int32_t>>, 2>
mesh::compute_connectivity(const Topology& topology, std::array<int, 2> d0,
    std::array<int, 2> d1)
{
    spdlog::info("Requesting connectivity ({}, {}) - ({}, {})", d0[0], d0[1],
        d1[0], d1[1]);

    // Already computed?
    if (topology.connectivity(d0, d1))
        return {nullptr, nullptr};

    // Same dimension, different entity types: no connectivity.
    if (d0[0] == d1[0] and d0[1] != d1[1])
        return {nullptr, nullptr};

    // No connectivity between certain entity type pairs.
    const CellType c0 = topology.entity_types(d0[0])[d0[1]];
    const CellType c1 = topology.entity_types(d1[0])[d1[1]];
    if ((c0 == CellType::hexahedron and c1 == CellType::triangle)
        or (c0 == CellType::triangle and c1 == CellType::hexahedron)
        or (c0 == CellType::tetrahedron and c1 == CellType::quadrilateral)
        or (c0 == CellType::quadrilateral and c1 == CellType::tetrahedron)) {
        return {nullptr, nullptr};
    }

    std::shared_ptr<const graph::AdjacencyList<std::int32_t>> c_d0_0
        = topology.connectivity(d0, {0, 0});
    if (d0[0] > 0 and !c_d0_0)
        throw std::runtime_error(std::format(
            "Missing entities of dimension {}.", d0[0]));

    std::shared_ptr<const graph::AdjacencyList<std::int32_t>> c_d1_0
        = topology.connectivity(d1, {0, 0});
    if (d1[0] > 0 and !c_d1_0)
        throw std::runtime_error(std::format(
            "Missing entities of dimension {}.", d1[0]));

    common::Timer timer(std::format("Compute connectivity {}-{}", d0[0], d0[1]));

    if (d0 == d1)
        return {std::make_shared<graph::AdjacencyList<std::int32_t>>(
                    c_d0_0->num_nodes()),
            nullptr};
    else if (d0[0] < d1[0]) {
        // Compute the d1 -> d0 connectivity (if needed) and transpose.
        if (!topology.connectivity(d1, d0)) {
            // Only possible case is edge->facet.
            assert(d0[0] == 1 and d1[0] == 2);
            auto c_d1_d0 = std::make_shared<graph::AdjacencyList<std::int32_t>>(
                compute_from_map(*c_d1_0, *c_d0_0));
            auto c_d0_d1 = std::make_shared<graph::AdjacencyList<std::int32_t>>(
                compute_from_transpose(*c_d1_d0, c_d0_0->num_nodes()));
            return {c_d0_d1, c_d1_d0};
        }
        else {
            assert(c_d0_0);
            auto c_d0_d1 = std::make_shared<graph::AdjacencyList<std::int32_t>>(
                compute_from_transpose(
                    *topology.connectivity(d1, d0), c_d0_0->num_nodes()));
            return {c_d0_d1, nullptr};
        }
    }
    else if (d0[0] > d1[0]) {
        // Compute by mapping vertices of a lower-dimensional entity to
        // those of a higher-dimensional one. Only possible case is
        // facet->edge.
        assert(d0[0] == 2 and d1[0] == 1);
        auto c_d0_d1 = std::make_shared<graph::AdjacencyList<std::int32_t>>(
            compute_from_map(*c_d0_0, *c_d1_0));
        return {c_d0_d1, nullptr};
    }
    else
        throw std::runtime_error(
            "Entity dimension error when computing topology.");
}
//-----------------------------------------------------------------------------
