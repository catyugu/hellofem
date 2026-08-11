// hellofem::fem — degree-of-freedom map on a mesh
// SPDX-License-Identifier: MIT

#include "DofMap.h"

#include "ElementDofLayout.h"
#include "common/IndexMap.h"
#include "common/sort.h"
#include "dofmapbuilder.h"
#include "graph/AdjacencyList.h"
#include "graph/ordering.h"
#include "mesh/Topology.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace hellofem::fem {

    namespace {

        /// Build a collapsed DofMap from a dofmap view, extracting dofs
        /// without building a new re-ordered dofmap.
        DofMap build_collapsed_dofmap(const DofMap& dofmap_view)
        {
            if (dofmap_view.element_dof_layout().block_size() > 1) {
                throw std::runtime_error(
                    "Cannot collapse a dofmap view with block size greater "
                    "than 1 when the parent has a block size of 1. Create new "
                    "dofmap first.");
            }

            // Build the set of dofs in the new (un-blocked) dofmap
            auto dofs_view_md = dofmap_view.map();
            std::vector<std::int32_t> dofs_view(
                dofs_view_md.data_handle(),
                dofs_view_md.data_handle() + dofs_view_md.size());
            hellofem::radix_sort(dofs_view);
            auto [unique_end, range_end] = std::ranges::unique(dofs_view);
            dofs_view.erase(unique_end, range_end);

            const int bs_view = dofmap_view.index_map_bs();

            // Create a sub-index map for the dofs
            std::shared_ptr<common::IndexMap> index_map;
            std::vector<std::int32_t> sub_imap_to_imap;
            if (bs_view == 1) {
                auto [_index_map, _sub_imap_to_imap]
                    = common::create_sub_index_map(*dofmap_view.index_map,
                        dofs_view, common::IndexMapOrder::preserve);
                index_map
                    = std::make_shared<common::IndexMap>(std::move(_index_map));
                sub_imap_to_imap = std::move(_sub_imap_to_imap);
            }
            else {
                std::vector<std::int32_t> indices;
                indices.reserve(dofs_view.size());
                std::ranges::transform(dofs_view,
                    std::back_inserter(indices),
                    [bs_view](auto idx) { return idx / bs_view; });
                auto [_index_map, _sub_imap_to_imap]
                    = common::create_sub_index_map(*dofmap_view.index_map,
                        indices, common::IndexMapOrder::preserve);
                index_map
                    = std::make_shared<common::IndexMap>(std::move(_index_map));
                sub_imap_to_imap = std::move(_sub_imap_to_imap);
            }

            // Map old dofs to new collapsed dofs
            const std::size_t array_size
                = dofs_view.empty() ? 0 : dofs_view.back() + bs_view;
            std::vector<std::int32_t> old_to_new(array_size, -1);
            for (std::size_t new_idx = 0; new_idx < sub_imap_to_imap.size();
                ++new_idx) {
                for (int k = 0; k < bs_view; ++k) {
                    const std::int32_t old_idx
                        = sub_imap_to_imap[new_idx] * bs_view + k;
                    assert(old_idx < (int)old_to_new.size());
                    old_to_new[old_idx] = new_idx;
                }
            }

            // Map dofs to new collapsed indices
            auto dof_array_view = dofmap_view.map();
            std::vector<std::int32_t> dofmap;
            dofmap.reserve(dof_array_view.size());
            std::ranges::transform(dof_array_view.data_handle(),
                dof_array_view.data_handle() + dof_array_view.size(),
                std::back_inserter(dofmap),
                [&old_to_new](auto idx_old) { return old_to_new[idx_old]; });

            // Copy the element layout, discarding parent data
            ElementDofLayout element_dof_layout
                = dofmap_view.element_dof_layout().copy();

            return DofMap(std::move(element_dof_layout), index_map, 1,
                std::move(dofmap), 1);
        }

    } // namespace

    graph::AdjacencyList<std::int32_t> transpose_dofmap(
        md::mdspan<const std::int32_t, md::dextents<std::size_t, 2>> dofmap,
        std::int32_t num_cells)
    {
        // Count the number of cell contributions to each global index
        const std::int32_t max_index = *std::max_element(dofmap.data_handle(),
            std::next(dofmap.data_handle(), num_cells * dofmap.extent(1)));

        std::vector<int> num_local_contributions(max_index + 1, 0);
        for (std::int32_t c = 0; c < num_cells; ++c) {
            auto dofs = md::submdspan(dofmap, c, md::full_extent);
            for (std::size_t d = 0; d < dofmap.extent(1); ++d)
                num_local_contributions[dofs[d]]++;
        }

        // Compute the offset for each global index
        std::vector<int> index_offsets(
            num_local_contributions.size() + 1, 0);
        std::partial_sum(num_local_contributions.begin(),
            num_local_contributions.end(), index_offsets.begin() + 1);

        std::vector<std::int32_t> data(index_offsets.back());
        std::vector<int> pos = index_offsets;
        int cell_offset = 0;
        for (std::int32_t c = 0; c < num_cells; ++c) {
            auto dofs = md::submdspan(dofmap, c, md::full_extent);
            for (std::size_t d = 0; d < dofmap.extent(1); ++d)
                data[pos[dofs[d]]++] = cell_offset++;
        }

        // `data` is already sorted per index: cell_offset only increases.
        return graph::AdjacencyList(std::move(data), std::move(index_offsets));
    }

    bool DofMap::operator==(const DofMap& map) const
    {
        return _index_map_bs == map._index_map_bs
            and _dofmap == map._dofmap and _bs == map._bs;
    }

    int DofMap::bs() const noexcept { return _bs; }

    DofMap DofMap::extract_sub_dofmap(std::span<const int> component) const
    {
        assert(!component.empty());

        // Components in the parent map that correspond to sub-dofs
        const std::vector sub_element_map_view
            = this->element_dof_layout().sub_view(component);

        const std::int32_t num_cells = this->_dofmap.size() / this->_shape1;
        const std::int32_t dofs_per_cell = sub_element_map_view.size();
        std::vector<std::int32_t> dofmap(num_cells * dofs_per_cell);
        const int bs_parent = this->bs();

        // Split each entry once, rather than per cell below
        std::vector<std::int32_t> parent_dof_index(dofs_per_cell);
        std::vector<std::int32_t> parent_dof_component(dofs_per_cell);
        for (std::int32_t i = 0; i < dofs_per_cell; ++i) {
            std::div_t pos = std::div(sub_element_map_view[i], bs_parent);
            parent_dof_index[i] = pos.quot;
            parent_dof_component[i] = pos.rem;
        }

        for (std::int32_t c = 0; c < num_cells; ++c) {
            auto cell_dmap_parent = this->cell_dofs(c);
            for (std::int32_t i = 0; i < dofs_per_cell; ++i) {
                dofmap[c * dofs_per_cell + i]
                    = bs_parent * cell_dmap_parent[parent_dof_index[i]]
                    + parent_dof_component[i];
            }
        }

        ElementDofLayout sub_dof_layout
            = _element_dof_layout.sub_layout(component);
        return DofMap(std::move(sub_dof_layout), this->index_map,
            this->index_map_bs(), std::move(dofmap), 1);
    }

    std::pair<DofMap, std::vector<std::int32_t>> DofMap::collapse(
        const mesh::Topology& topology,
        std::function<std::vector<int>(
            const graph::AdjacencyList<std::int32_t>&)>&& reorder_fn) const
    {
        if (!reorder_fn) {
            // graph::reorder_rcm returns vector<int32>; the reorder hook
            // expects vector<int>.
            reorder_fn = [](const graph::AdjacencyList<std::int32_t>& graph) {
                std::vector<int> remap = graph::reorder_rcm(graph);
                return remap;
            };
        }

        // Create the new dofmap: rebuild a blocked dofmap from scratch if
        // the parent has no block structure but the sub-map does; otherwise
        // collapse the view directly.
        DofMap dofmap_new = [&]() -> DofMap {
            if (index_map_bs() == 1
                and _element_dof_layout.block_size() > 1) {
                ElementDofLayout collapsed_dof_layout
                    = _element_dof_layout.copy();
                auto [_index_map, bs, dofmaps] = build_dofmap_data(topology,
                    {collapsed_dof_layout}, reorder_fn);
                auto index_map
                    = std::make_shared<common::IndexMap>(std::move(_index_map));
                return DofMap(_element_dof_layout, index_map, bs,
                    std::move(dofmaps.front()), bs);
            }
            else {
                return build_collapsed_dofmap(*this);
            }
        }();

        // Map from collapsed dof index to original dof index
        auto index_map_new = dofmap_new.index_map;
        const std::int32_t size
            = (index_map_new->size_local() + index_map_new->num_ghosts())
            * dofmap_new.index_map_bs();
        std::vector<std::int32_t> collapsed_map(size);

        const int bs = dofmap_new.bs();
        const std::size_t num_cells = _dofmap.size() / _shape1;
        for (std::size_t c = 0; c < num_cells; ++c) {
            std::span<const std::int32_t> cell_dofs_view = this->cell_dofs(c);
            std::span<const std::int32_t> cell_dofs = dofmap_new.cell_dofs(c);
            for (std::size_t i = 0; i < cell_dofs.size(); ++i) {
                for (int k = 0; k < bs; ++k) {
                    assert(bs * cell_dofs[i] + k < (int)collapsed_map.size());
                    assert(bs * i + k < (int)cell_dofs_view.size());
                    collapsed_map[bs * cell_dofs[i] + k]
                        = cell_dofs_view[bs * i + k];
                }
            }
        }

        return {std::move(dofmap_new), std::move(collapsed_map)};
    }

    md::mdspan<const std::int32_t, md::dextents<std::size_t, 2>>
    DofMap::map() const
    {
        return md::mdspan<const std::int32_t,
            md::dextents<std::size_t, 2>>(
            _dofmap.data(), _dofmap.size() / _shape1, _shape1);
    }

    int DofMap::index_map_bs() const { return _index_map_bs; }

} // namespace hellofem::fem
