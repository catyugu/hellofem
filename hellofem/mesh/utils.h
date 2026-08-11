// hellofem::mesh — mesh utilities
// SPDX-License-Identifier: MIT

#pragma once

#include "Topology.h"

#include <cstdint>
#include <span>
#include <vector>

namespace hellofem::mesh {

    class Mesh;

    /// For each entity in `entities` (of dimension `d0`), collect the
    /// distinct incident entities of dimension `d1` via the `d0 -> d1`
    /// connectivity. Requires the entities of both dimensions and the
    /// connectivity to exist.
    std::vector<std::int32_t>
    compute_incident_entities(const Topology& topology,
        std::span<const std::int32_t> entities, int d0, int d1);

} // namespace hellofem::mesh
