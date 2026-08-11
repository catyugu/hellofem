// hellofem::mesh — compute entities and connectivity of a topology
// SPDX-License-Identifier: MIT

#pragma once

#include "cell_types.h"
#include "common/IndexMap.h"
#include "graph/AdjacencyList.h"

#include <array>
#include <cstdint>
#include <memory>
#include <tuple>
#include <vector>

namespace hellofem::mesh {

    class Topology;

    /// Compute the entities of dimension `dim` and type `entity_type`
    /// of a topology. Vertices (dim == 0) always exist and return empty
    /// data. If the entities already exist, empty data is returned.
    ///
    /// @return (cell-entity connectivity, entity-vertex connectivity,
    /// index map of the new entities, inter-process entity indices).
    std::tuple<std::vector<std::shared_ptr<graph::AdjacencyList<std::int32_t>>>,
        std::shared_ptr<graph::AdjacencyList<std::int32_t>>,
        std::shared_ptr<common::IndexMap>, std::vector<std::int32_t>>
    compute_entities(const Topology& topology, int dim, CellType entity_type,
        int num_threads);

    /// Compute the connectivity between two entity types `d0` and `d1`,
    /// each given as a pair (dimension, entity-type index). Requires the
    /// entities of both dimensions to exist.
    ///
    /// @return The `(d0, d1)` connectivity and, if it was computed as a
    /// side product, the `(d1, d0)` connectivity.
    std::array<std::shared_ptr<graph::AdjacencyList<std::int32_t>>, 2>
    compute_connectivity(const Topology& topology, std::array<int, 2> d0,
        std::array<int, 2> d1);

} // namespace hellofem::mesh
