// hellofem::mesh — compute entity permutations and reflections
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <utility>
#include <vector>

namespace hellofem::mesh {

    class Topology;

    /// Compute the entity permutations and reflections for a mesh.
    ///
    /// @return (facet permutations, cell permutation info). The facet
    /// permutations are stored flattened as
    /// `data[cell * facets_per_cell + facet]` with `data[..] % 2` the
    /// number of reflections and `data[..] / 2` the number of rotations.
    /// The cell permutation info encodes, per cell, the rotations and
    /// reflections of every facet and edge of the cell (see
    /// Topology::get_cell_permutation_info).
    std::pair<std::vector<std::uint8_t>, std::vector<std::uint32_t>>
    compute_entity_permutations(const Topology& topology, int num_threads);

} // namespace hellofem::mesh
