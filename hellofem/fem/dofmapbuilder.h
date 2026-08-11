// hellofem::fem — build dofmap data on a mesh topology
// SPDX-License-Identifier: MIT

#pragma once

#include "graph/AdjacencyList.h"

#include <cstdint>
#include <functional>
#include <tuple>
#include <vector>

namespace hellofem::common {
    class IndexMap;
}

namespace hellofem::mesh {
    class Topology;
}

namespace hellofem::fem {

    class ElementDofLayout;
    class DofMap;

    /// Build dofmap data for elements on a mesh topology.
    ///
    /// @param[in] topology The mesh topology.
    /// @param[in] element_dof_layouts The element dof layouts for each
    /// cell type in `topology`.
    /// @param[in] reorder_fn Graph reordering function applied to the
    /// dofmaps (or nullptr for none).
    /// @return The index map, block size, and dofmaps for each element
    /// type.
    std::tuple<common::IndexMap, int, std::vector<std::vector<std::int32_t>>>
    build_dofmap_data(const mesh::Topology& topology,
        const std::vector<ElementDofLayout>& element_dof_layouts,
        const std::function<std::vector<int>(
            const graph::AdjacencyList<std::int32_t>&)>& reorder_fn
        = nullptr);

} // namespace hellofem::fem
