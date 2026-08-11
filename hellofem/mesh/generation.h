// hellofem::mesh — programmatic mesh generation
// SPDX-License-Identifier: MIT

#pragma once

#include "Geometry.h"
#include "Mesh.h"
#include "Topology.h"
#include "mesh/cell_types.h"

#include <concepts>
#include <memory>

namespace hellofem::mesh {

    /// Create a mesh on the unit square `[0, 1]^2` divided into `n`
    /// intervals per edge, using `2 n^2` triangles with an affine P1 map.
    ///
    /// The diagonal of each square runs from the bottom-left to the
    /// top-right corner; vertex numbering follows `v(i, j) = i + j*(n+1)`.
    ///
    /// @param[in] n Number of intervals per edge (>= 1).
    /// @return A mesh with linear (P1) geometry.
    std::shared_ptr<Mesh<double>> create_unit_square(int n);

} // namespace hellofem::mesh
