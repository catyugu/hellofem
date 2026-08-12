// hellofem::io — cell node ordering for mesh input/output
// SPDX-License-Identifier: MIT

#pragma once

#include "mesh/cell_types.h"

#include <array>
#include <cstdint>
#include <span>
#include <tuple>
#include <vector>

/// Node-order permutations between the basix reference ordering (used by
/// the mesh geometry) and the VTK file ordering, plus VTK cell type
/// identifiers.
///
/// For a cell with `n` nodes, `p = perm_vtk(type, n)` maps from the VTK
/// ordering to the basix ordering: `a_basix[i] = a_vtk[p[i]]`. The VTK
/// Lagrange node orderings follow the published VTK conventions:
///
/// @verbatim
/// Triangle:               Tetrahedron:       Quadrilateral:   Hexahedron:
///       2                    3                   3-----2           6------7
///      / \                  / \                /      |          /|     /|
///     /   \                /   \              0      |         | |    | |
///    4     5              7     5             |      |         4------5 |
///   /       \            /       \            |      1         | 2----|-3
///  0-----3---1          2---6-4---0           |     /          |/     |/
///                      /           \          |    /           0------1
///                     /             \         0---1
///                    8------9--------1
/// @endverbatim
///
/// (Edge and interior nodes are numbered in the order they appear in the
/// file; see the individual permutations for the exact mapping.)

namespace hellofem::io::cells {

    /// Lagrange degree of a cell with `num_nodes` nodes.
    ///
    /// @param[in] type Cell shape.
    /// @param[in] num_nodes Number of nodes of the cell.
    int cell_degree(mesh::CellType type, int num_nodes);

    /// Permutation from the VTK node ordering to the basix ordering.
    ///
    /// @return Array `p` with `a_basix[i] = a_vtk[p[i]]`.
    std::vector<std::uint16_t> perm_vtk(mesh::CellType type, int num_nodes);

    /// Transpose of a re-ordering map (`map` with each entry replaced by
    /// its position).
    std::vector<std::uint16_t> transpose(std::span<const std::uint16_t> map);

    /// Apply a permutation to each cell of a topology array.
    ///
    /// @param[in] cells Cell topologies (row-major, `shape`).
    /// @param[in] shape Shape of `cells`.
    /// @param[in] p Permutation with `a_p[i] = a[p[i]]`.
    /// @return Permuted cell array, same shape as `cells`.
    std::vector<std::int64_t> apply_permutation(
        std::span<const std::int64_t> cells, std::array<std::size_t, 2> shape,
        std::span<const std::uint16_t> p);

    /// VTK cell type identifier for a cell shape.
    std::int8_t get_vtk_cell_type(mesh::CellType cell);

    /// Cell type and degree from a VTK cell type identifier.
    ///
    /// @return Cell type, and -1 for the polynomial degree when the VTK
    /// identifier encodes an arbitrary-order Lagrange cell.
    std::tuple<mesh::CellType, std::int8_t> vtk_to_dolfinx(std::int8_t vtk_type);

} // namespace hellofem::io::cells
