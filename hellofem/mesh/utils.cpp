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

} // namespace hellofem::mesh
