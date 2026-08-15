// hellofem::fem — shared assembly utilities
// SPDX-License-Identifier: MIT

#pragma once

#include "mesh/Mesh.h"
#include "mesh/Topology.h"
#include "mesh/utils.h"

#include <algorithm>
#include <cstdint>
#include <span>
#include <vector>

namespace hellofem::fem {

    /// Flattened `(cell, local_facet)` pairs for the exterior facets of a
    /// mesh. Each exterior facet is encoded as the single cell it is
    /// attached to together with the facet's local index in that cell.
    std::vector<std::int32_t> exterior_facet_entities(const mesh::Topology& topology);

    /// Flattened `(cell+, local_facet+, cell-, local_facet-)` quadruples
    /// for the interior facets of a mesh (every facet shared by exactly
    /// two cells).
    std::vector<std::int32_t> interior_facet_entities(const mesh::Topology& topology);

} // namespace hellofem::fem
