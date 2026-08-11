// hellofem::fem — degree-of-freedom map on a mesh
// SPDX-License-Identifier: MIT

#pragma once

#include "ElementDofLayout.h"
#include "common/types.h"
#include "graph/AdjacencyList.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace hellofem::common {
    class IndexMap;
}

namespace hellofem::mesh {
    class Topology;
}

namespace hellofem::fem {

    /// Transpose a dofmap: for each global dof, the cells that contain it.
    graph::AdjacencyList<std::int32_t> transpose_dofmap(
        md::mdspan<const std::int32_t, md::dextents<std::size_t, 2>> dofmap,
        std::int32_t num_cells);

    /// Map from cell dofs to global dofs (single-process).
    ///
    /// A dofmap holds, for each cell, the global dof indices of the
    /// degrees-of-freedom on that cell. Dof indices are scalar-block
    /// indices into the shared `index_map`; `bs` describes the number of
    /// physical dofs co-located at each block index.
    class DofMap {
    public:
        /// Create a dofmap.
        /// @param[in] element Element dof layout.
        /// @param[in] index_map Index map for the dofs.
        /// @param[in] index_map_bs Block size of `index_map`.
        /// @param[in] dofmap Cell dofs, row-major `(num_cells, dofs_per_cell)`.
        /// @param[in] bs Block size of the dofmap entries.
        template <typename E, typename U>
            requires std::is_convertible_v<std::remove_cvref_t<E>,
                         fem::ElementDofLayout>
                         and std::is_convertible_v<std::remove_cvref_t<U>,
                             std::vector<std::int32_t>>
        DofMap(E&& element, std::shared_ptr<const common::IndexMap> index_map,
            int index_map_bs, U&& dofmap, int bs)
            : index_map(std::move(index_map)), _index_map_bs(index_map_bs), _element_dof_layout(element), _dofmap(std::forward<U>(dofmap)), _bs(bs), _shape1(_element_dof_layout.num_dofs() * _element_dof_layout.block_size() / _bs)
        {
        }

        /// Copy constructor
        DofMap(const DofMap& map) = delete;

        /// Move constructor
        DofMap(DofMap&& map) = default;

        /// Destructor
        ~DofMap() = default;

        /// Move assignment
        DofMap& operator=(DofMap&& map) = default;

        /// Equality operator (compares index-map block size, dofmap data
        /// and block size).
        bool operator==(const DofMap& map) const;

        /// Global dofs for cell `c`.
        std::span<const std::int32_t> cell_dofs(std::int32_t c) const
        {
            return std::span(_dofmap.data() + _shape1 * c, _shape1);
        }

        /// Block size of the dofmap entries.
        int bs() const noexcept;

        /// Extract a sub-dofmap for a component of the element.
        DofMap extract_sub_dofmap(std::span<const int> component) const;

        /// Collapse a dofmap view (or build a new one) so that it owns
        /// exactly its dofs, numbered contiguously.
        /// @param[in] topology The mesh topology (needed to rebuild a
        /// blocked dofmap).
        /// @param[in] reorder_fn Graph reordering applied when rebuilding.
        /// @return The collapsed dofmap and a map from new (collapsed)
        /// dof indices to the original dof indices.
        std::pair<DofMap, std::vector<std::int32_t>> collapse(
            const mesh::Topology& topology,
            std::function<std::vector<int>(
                const graph::AdjacencyList<std::int32_t>&)>&& reorder_fn
            = nullptr) const;

        /// The dofmap as a 2D mdspan `(num_cells, dofs_per_cell)`.
        md::mdspan<const std::int32_t, md::dextents<std::size_t, 2>>
        map() const;

        /// Element dof layout.
        const ElementDofLayout& element_dof_layout() const
        {
            return _element_dof_layout;
        }

        /// Index map for the dofs.
        std::shared_ptr<const common::IndexMap> index_map;

        /// Block size of the index map.
        int index_map_bs() const;

    private:
        // Block size of the index map
        int _index_map_bs = -1;

        // Element dof layout
        ElementDofLayout _element_dof_layout;

        // Flat dofmap, row-major (cell, dof)
        std::vector<std::int32_t> _dofmap;

        // Block size of the dofmap entries
        int _bs = -1;

        // Number of dofs per cell row
        int _shape1 = -1;
    };

} // namespace hellofem::fem
