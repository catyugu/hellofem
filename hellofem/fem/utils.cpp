// hellofem::fem — shared assembly utilities
// SPDX-License-Identifier: MIT

#include "utils.h"

#include "graph/AdjacencyList.h"

#include <algorithm>
#include <cstdint>
#include <span>
#include <vector>

namespace hellofem::fem {

    namespace {

        /// Local index of entity `e` within cell `c`: the position of `e`
        /// in the cell's `c_to_e` list.
        int local_entity_index(const graph::AdjacencyList<std::int32_t>& c_to_e,
            std::int32_t c, std::int32_t e)
        {
            auto cell_entities = c_to_e.links(c);
            auto it = std::find(cell_entities.begin(), cell_entities.end(), e);
            return static_cast<int>(std::distance(cell_entities.begin(), it));
        }

    } // namespace

    std::vector<std::int32_t> exterior_facet_entities(const mesh::Topology& topology)
    {
        const int tdim = topology.dim();
        auto* mutable_topo = const_cast<mesh::Topology*>(&topology);
        mutable_topo->create_entities(tdim - 1);
        mutable_topo->create_connectivity(tdim - 1, tdim);
        mutable_topo->create_connectivity(tdim, tdim - 1);

        const std::vector<std::int32_t> facets
            = mesh::exterior_facet_indices(topology);
        auto e_to_c = topology.connectivity(tdim - 1, tdim);
        auto c_to_e = topology.connectivity(tdim, tdim - 1);

        std::vector<std::int32_t> entities;
        entities.reserve(2 * facets.size());
        for (std::int32_t f : facets) {
            auto cells = e_to_c->links(f);
            assert(!cells.empty());
            const std::int32_t c = cells.front();
            entities.push_back(c);
            entities.push_back(local_entity_index(*c_to_e, c, f));
        }
        return entities;
    }

    std::vector<std::int32_t> interior_facet_entities(const mesh::Topology& topology)
    {
        const int tdim = topology.dim();
        auto* mutable_topo = const_cast<mesh::Topology*>(&topology);
        mutable_topo->create_entities(tdim - 1);
        mutable_topo->create_connectivity(tdim - 1, tdim);
        mutable_topo->create_connectivity(tdim, tdim - 1);

        auto e_to_c = topology.connectivity(tdim - 1, tdim);
        auto c_to_e = topology.connectivity(tdim, tdim - 1);
        const std::int32_t num_facets = e_to_c->num_nodes();

        std::vector<std::int32_t> entities;
        for (std::int32_t f = 0; f < num_facets; ++f) {
            auto cells = e_to_c->links(f);
            if (cells.size() != 2)
                continue;
            entities.push_back(cells[0]);
            entities.push_back(local_entity_index(*c_to_e, cells[0], f));
            entities.push_back(cells[1]);
            entities.push_back(local_entity_index(*c_to_e, cells[1], f));
        }
        return entities;
    }

} // namespace hellofem::fem
