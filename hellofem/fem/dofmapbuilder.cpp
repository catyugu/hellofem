// hellofem::fem — build dofmap data on a mesh topology
// SPDX-License-Identifier: MIT

#include "dofmapbuilder.h"

#include "ElementDofLayout.h"
#include "common/IndexMap.h"
#include "common/Timer.h"
#include "mesh/Topology.h"
#include "mesh/cell_types.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace hellofem;

namespace {

    /// dofmap as a flattened 2D array of fixed width.
    struct dofmap_t {
        std::int32_t width;
        std::vector<std::int32_t> array;
    };

    /// Build a dofmap from ElementDofLayouts based on mesh entity
    /// indices. Single-process: every entity is owned, so a dof index is
    /// uniquely determined by its (entity type, entity index, local
    /// dof) triple, and no ghost handling is needed.
    ///
    /// @return (dofmaps per cell type, number of local dofs).
    std::pair<std::vector<dofmap_t>, std::int32_t>
    build_basic_dofmaps(const mesh::Topology& topology,
        const std::vector<fem::ElementDofLayout>& element_dof_layouts)
    {
        common::Timer t0("Dofmap builder: init dofmap from element dofmap");

        const std::size_t D = topology.dim();
        const std::size_t num_cell_types = topology.entity_types(D).size();

        // Find which (dimension, entity type) combinations carry dofs,
        // the number of dofs per such entity, and the local offset at
        // which the block of that entity type starts.
        std::vector<std::pair<std::int8_t, std::int8_t>> required_dim_et;
        std::vector<std::int32_t> num_entity_dofs_et;
        std::vector<std::int32_t> local_entity_offsets {0};

        std::vector<std::vector<mesh::CellType>> entity_types(D + 1);
        for (std::size_t d = 0; d <= D; ++d)
            entity_types[d] = topology.entity_types(d);

        for (std::size_t i = 0; i < num_cell_types; ++i) {
            const mesh::CellType cell_type = entity_types[D][i];
            const auto& entity_dofs = element_dof_layouts[i].entity_dofs_all();

            for (std::size_t d = 0; d <= D; ++d) {
                const auto& entity_dofs_d = entity_dofs[d];
                for (std::size_t e = 0; e < entity_dofs_d.size(); ++e) {
                    if (entity_dofs_d[e].empty())
                        continue;

                    const auto et_it = std::find(entity_types[d].begin(),
                        entity_types[d].end(),
                        mesh::cell_entity_type(cell_type, d, e));
                    assert(et_it != entity_types[d].end());
                    const int et_index
                        = std::ranges::distance(entity_types[d].begin(), et_it);

                    auto required_entity_it = std::find(required_dim_et.begin(),
                        required_dim_et.end(),
                        std::pair<std::int8_t, std::int8_t> {d, et_index});
                    if (required_entity_it == required_dim_et.end()) {
                        required_dim_et.push_back({d, et_index});
                        const std::int32_t num_entity_dofs
                            = entity_dofs_d[e].size();
                        num_entity_dofs_et.push_back(num_entity_dofs);
                        auto im = topology.index_maps(d)[et_index];
                        local_entity_offsets.push_back(
                            local_entity_offsets.back()
                            + num_entity_dofs
                                * (im->size_local() + im->num_ghosts()));
                        if (d < D
                            and !topology.connectivity({int(D), int(i)},
                                {int(d), et_index}))
                        {
                            throw std::runtime_error(std::format(
                                "Missing needed connectivity. Cell type: {} to "
                                "dim: {}, ent: {}",
                                i, d, et_index));
                        }
                    }
                    else {
                        const std::size_t k
                            = std::ranges::distance(required_dim_et.begin(),
                                required_entity_it);
                        if (num_entity_dofs_et[k]
                            != (int)entity_dofs_d[e].size())
                            throw std::runtime_error(
                                "Incompatible elements detected.");
                    }
                }
            }
        }

        // Build the dofmap for each cell type.
        std::vector<dofmap_t> dofs(num_cell_types);
        for (std::size_t i = 0; i < num_cell_types; ++i) {
            const mesh::CellType cell_type = entity_types[D][i];
            const auto entity_dofs = element_dof_layouts[i].entity_dofs_all();
            assert(entity_dofs.size() == D + 1);

            const std::int32_t num_cells
                = topology.index_maps(D)[i]->size_local()
                + topology.index_maps(D)[i]->num_ghosts();
            const std::int32_t dofmap_width = element_dof_layouts[i].num_dofs();
            dofs[i].width = dofmap_width;
            dofs[i].array.resize(num_cells * dofmap_width);

            std::vector<std::vector<mesh::CellType>> cell_entity_types(D + 1);
            for (std::size_t d = 0; d < D + 1; ++d) {
                const int entities_d = mesh::cell_num_entities(cell_type, d);
                cell_entity_types[d].reserve(entities_d);
                for (int e = 0; e < entities_d; ++e)
                    cell_entity_types[d].push_back(
                        mesh::cell_entity_type(cell_type, d, e));
            }

            // Per-entity data (cell-invariant), precomputed once.
            struct required_entity_data {
                std::size_t d;
                std::int32_t local_offset;
                int num_entity_dofs_expected;
                const std::vector<std::vector<int>>* e_dofs_d;
                const std::vector<mesh::CellType>* e_types;
                mesh::CellType e_type;
                std::shared_ptr<const graph::AdjacencyList<std::int32_t>>
                    connectivity;
            };
            std::vector<required_entity_data> required_entities;
            required_entities.reserve(required_dim_et.size());
            for (std::size_t k = 0; k < required_dim_et.size(); ++k) {
                const std::size_t d = required_dim_et[k].first;
                const std::size_t et = required_dim_et[k].second;

                // Skip undefined topology, e.g. quad facets of tetrahedra.
                auto c = d < D ? topology.connectivity({int(D), int(i)},
                                     {int(d), int(et)})
                               : nullptr;
                if (d < D and !c)
                    continue;

                required_entities.push_back({d, local_entity_offsets[k],
                    num_entity_dofs_et[k], &entity_dofs[d],
                    &cell_entity_types[d], topology.entity_types(d)[et], c});
            }

            std::int32_t dofmap_offset = 0;
            for (std::int32_t c = 0; c < num_cells; ++c) {
                std::span<std::int32_t> dofs_c(
                    dofs[i].array.data() + dofmap_offset, dofmap_width);
                dofmap_offset += dofmap_width;

                for (const required_entity_data& re : required_entities) {
                    std::span<const std::int32_t> c_to_e
                        = re.connectivity
                        ? re.connectivity->links(c)
                        : std::span<const std::int32_t>(&c, 1);

                    const auto& e_dofs_d = *re.e_dofs_d;
                    const auto& e_types = *re.e_types;
                    int w = 0;
                    for (std::size_t e = 0; e < e_dofs_d.size(); ++e) {
                        // Skip entities of wrong type (e.g. for facets of
                        // a prism).
                        if (re.e_type != e_types[e])
                            continue;

                        const auto& e_dofs_d_e = e_dofs_d[e];
                        assert((int)e_dofs_d_e.size()
                            == re.num_entity_dofs_expected);
                        const std::int32_t e_index_local = c_to_e[w];
                        ++w;

                        for (std::size_t j = 0; j < e_dofs_d_e.size(); ++j)
                            dofs_c[e_dofs_d_e[j]]
                                = re.local_offset
                                + re.num_entity_dofs_expected * e_index_local
                                + j;
                    }
                }
            }
        }

        return {std::move(dofs), local_entity_offsets.back()};
    }

    /// Re-order dofs for locality by iterating over cells, and apply an
    /// optional graph reordering to the owned dofs.
    ///
    /// @return The old-to-new local index map.
    std::vector<std::int32_t> compute_reordering_map(
        const std::vector<dofmap_t>& dofmaps,
        const std::function<std::vector<int>(
            const graph::AdjacencyList<std::int32_t>&)>& reorder_fn)
    {
        std::int32_t local_size = 0;
        for (auto& dofmap : dofmaps)
            for (auto d : dofmap.array)
                local_size = std::max(local_size, d + 1);

        std::vector<int> original_to_contiguous(local_size, -1);
        std::int32_t counter = 0;
        for (auto& dofmap : dofmaps)
            for (std::int32_t dof : dofmap.array)
                if (original_to_contiguous[dof] == -1)
                    original_to_contiguous[dof] = counter++;

        // Any dofs not appearing in a cell go at the end.
        for (std::size_t dof = 0; dof < original_to_contiguous.size(); ++dof)
            if (original_to_contiguous[dof] == -1)
                original_to_contiguous[dof] = counter++;

        if (reorder_fn) {
            // Build the dof adjacency graph and apply the reordering.
            std::vector<std::int32_t> num_edges(local_size, 0);
            for (auto& dofmap : dofmaps) {
                const std::size_t num_cells = dofmap.array.size() / dofmap.width;
                for (std::size_t cell = 0; cell < num_cells; ++cell) {
                    std::vector<std::int32_t> node_temp;
                    for (std::int32_t i = 0; i < dofmap.width; ++i) {
                        const std::int32_t node = original_to_contiguous
                            [dofmap.array[cell * dofmap.width + i]];
                        node_temp.push_back(node);
                    }
                    for (std::int32_t node : node_temp)
                        num_edges[node] += node_temp.size() - 1;
                }
            }

            std::vector<std::int32_t> offsets(num_edges.size() + 1, 0);
            std::partial_sum(num_edges.begin(), num_edges.end(),
                std::next(offsets.begin(), 1));
            std::vector<std::int32_t> edges(offsets.back());
            std::vector<std::int32_t> counter_offset(num_edges.size(), 0);
            for (auto& dofmap : dofmaps) {
                const std::size_t num_cells = dofmap.array.size() / dofmap.width;
                std::vector<std::int32_t> node_temp;
                for (std::size_t cell = 0; cell < num_cells; ++cell) {
                    node_temp.clear();
                    for (std::int32_t i = 0; i < dofmap.width; ++i)
                        node_temp.push_back(
                            original_to_contiguous
                                [dofmap.array[cell * dofmap.width + i]]);
                    for (std::size_t i = 0; i < node_temp.size(); ++i) {
                        const std::int32_t node_0 = node_temp[i];
                        for (std::size_t j = i + 1; j < node_temp.size(); ++j) {
                            const std::int32_t node_1 = node_temp[j];
                            edges[offsets[node_0] + counter_offset[node_0]++]
                                = node_1;
                            edges[offsets[node_1] + counter_offset[node_1]++]
                                = node_0;
                        }
                    }
                }
            }

            // Eliminate duplicate edges.
            std::vector<std::int32_t> graph_data, graph_offsets {0};
            std::int32_t current_offset = 0;
            for (std::size_t i = 0; i < num_edges.size(); ++i) {
                auto range_begin = std::next(edges.begin(), current_offset);
                auto edge_range = std::ranges::subrange(
                    range_begin, std::next(range_begin, num_edges[i]));
                std::ranges::sort(edge_range);
                auto it = std::ranges::unique(edge_range).begin();
                graph_data.insert(graph_data.end(), range_begin, it);
                graph_offsets.push_back(graph_offsets.back()
                    + std::ranges::distance(range_begin, it));
                current_offset += num_edges[i];
            }

            const std::vector<int> node_remap = reorder_fn(
                graph::AdjacencyList(std::move(graph_data),
                    std::move(graph_offsets)));
            std::ranges::transform(original_to_contiguous,
                original_to_contiguous.begin(),
                [&node_remap](auto index) {
                    return index < (int)node_remap.size()
                        ? node_remap[index]
                        : index;
                });
        }

        return original_to_contiguous;
    }

} // namespace

//-----------------------------------------------------------------------------
std::tuple<common::IndexMap, int, std::vector<std::vector<std::int32_t>>>
fem::build_dofmap_data(const mesh::Topology& topology,
    const std::vector<ElementDofLayout>& element_dof_layouts,
    const std::function<std::vector<int>(
        const graph::AdjacencyList<std::int32_t>&)>& reorder_fn)
{
    common::Timer t0("Dofmap builder: build dofmap data");

    const auto [node_graphs, local_size]
        = build_basic_dofmaps(topology, element_dof_layouts);

    const std::vector<std::int32_t> old_to_new
        = compute_reordering_map(node_graphs, reorder_fn);

    // Single-process: every dof is owned, numbered 0..local_size-1
    // globally.
    common::IndexMap index_map(0, local_size);

    // Build re-ordered dofmaps.
    std::vector<std::vector<std::int32_t>> dofmaps(node_graphs.size());
    for (std::size_t i = 0; i < dofmaps.size(); ++i) {
        const std::vector<std::int32_t>& node_graphs_i = node_graphs[i].array;
        dofmaps[i].resize(node_graphs_i.size());
        std::vector<std::int32_t>& dofmaps_i = dofmaps[i];
        for (std::size_t j = 0; j < node_graphs_i.size(); ++j)
            dofmaps_i[j] = old_to_new[node_graphs_i[j]];
    }

    return {std::move(index_map), element_dof_layouts.front().block_size(),
        std::move(dofmaps)};
}
//-----------------------------------------------------------------------------
