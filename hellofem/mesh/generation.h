// hellofem::mesh — programmatic mesh generation
// SPDX-License-Identifier: MIT

#pragma once

#include "Geometry.h"
#include "Mesh.h"
#include "Topology.h"
#include "mesh/cell_types.h"

#include <array>
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

    /// Create a mesh on the rectangle `[p0, p1]` divided into `n`
    /// intervals per edge, using `2 nx ny` triangles with an affine P1
    /// map.
    ///
    /// @param[in] p0 Lower-left corner.
    /// @param[in] p1 Upper-right corner.
    /// @param[in] n Number of intervals per edge (each >= 1).
    std::shared_ptr<Mesh<double>> create_rectangle(
        std::array<double, 2> p0, std::array<double, 2> p1,
        std::array<int, 2> n);

    /// Create a box mesh on `[p0, p1]` divided into `n` intervals per
    /// edge, using `nx ny nz` hexahedra with an affine P1 map.
    ///
    /// @param[in] p0 Lower-back-left corner.
    /// @param[in] p1 Upper-front-right corner.
    /// @param[in] n Number of intervals per edge (each >= 1).
    std::shared_ptr<Mesh<double>> create_box(
        std::array<double, 3> p0, std::array<double, 3> p1,
        std::array<int, 3> n);

} // namespace hellofem::mesh
