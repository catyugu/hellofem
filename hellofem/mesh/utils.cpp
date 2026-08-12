// hellofem::mesh — mesh utilities
// SPDX-License-Identifier: MIT

#include "utils.h"

#include "Topology.h"

#include <algorithm>
#include <cstdint>
#include <format>
#include <stdexcept>
#include <vector>

namespace hellofem::mesh {

    std::vector<std::int32_t>
    compute_incident_entities(const Topology& topology,
        std::span<const std::int32_t> entities, int d0, int d1)
    {
        auto map0 = topology.index_map(d0);
        if (!map0)
            throw std::runtime_error(std::format(
                "Mesh entities of dimension {} have not been created.", d0));

        auto map1 = topology.index_map(d1);
        if (!map1)
            throw std::runtime_error(std::format(
                "Mesh entities of dimension {} have not been created.", d1));

        auto e0_to_e1 = topology.connectivity(d0, d1);
        if (!e0_to_e1)
            throw std::runtime_error(
                std::format("Connectivity missing: ({}, {})", d0, d1));

        std::vector<std::int32_t> entities1;
        for (std::int32_t entity : entities) {
            auto e = e0_to_e1->links(entity);
            entities1.insert(entities1.end(), e.begin(), e.end());
        }

        std::ranges::sort(entities1);
        auto [unique_end, range_end] = std::ranges::unique(entities1);
        entities1.erase(unique_end, range_end);

        return entities1;
    }

    std::vector<std::int32_t>
    exterior_facet_indices(const Topology& topology, int facet_type_idx)
    {
        const int tdim = topology.dim();
        if (facet_type_idx < 0
            or static_cast<std::size_t>(facet_type_idx)
                >= topology.entity_types(tdim - 1).size())
            throw std::runtime_error("Invalid facet type index.");

        auto f_to_c
            = topology.connectivity({tdim - 1, facet_type_idx}, {tdim, 0});
        if (!f_to_c)
            throw std::runtime_error(
                "Facet-to-cell connectivity has not been computed.");

        // A facet is exterior when exactly one cell is attached to it.
        // In the single-process build all facets are owned.
        std::vector<std::int32_t> facets;
        for (std::int32_t f = 0; f < f_to_c->num_nodes(); ++f)
            if (f_to_c->num_links(f) == 1)
                facets.push_back(f);

        return facets;
    }

    std::vector<std::int32_t>
    exterior_facet_indices(const Topology& topology)
    {
        const int tdim = topology.dim();
        if (topology.entity_types(tdim - 1).size() > 1)
            throw std::runtime_error("Multiple facet types in mesh. Call "
                                     "exterior_facet_indices with a facet type index.");

        return exterior_facet_indices(topology, 0);
    }

} // namespace hellofem::mesh
