// hellofem::io — cell node ordering for mesh input/output
// SPDX-License-Identifier: MIT

#include "cells.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <stdexcept>

using namespace hellofem;

namespace {

    /// Permutation arrays from the VTK ordering to the basix ordering,
    /// indexed by (cell, degree). These encode the published VTK Lagrange
    /// node orderings: for a cell with `n` nodes, `p` maps the VTK
    /// position `i` to the basix node `p[i]`.
    std::vector<std::uint16_t> vtk_interval(int num_nodes)
    {
        // Interval nodes are in the same order in both conventions.
        std::vector<std::uint16_t> map(num_nodes);
        std::iota(map.begin(), map.end(), 0);
        return map;
    }

    std::vector<std::uint16_t> vtk_triangle(int num_nodes)
    {
        switch (num_nodes) {
        case 3:
            return {0, 1, 2};
        case 6:
            return {0, 1, 2, 5, 3, 4};
        case 10:
            return {0, 1, 2, 7, 8, 3, 4, 6, 5, 9};
        default:
            throw std::runtime_error(
                "perm_vtk: unsupported triangle degree.");
        }
    }

    std::vector<std::uint16_t> vtk_tetrahedron(int num_nodes)
    {
        switch (num_nodes) {
        case 4:
            return {0, 1, 2, 3};
        case 10:
            return {0, 1, 2, 3, 9, 6, 8, 7, 5, 4};
        case 20:
            return {0, 1, 2, 3, 14, 15, 8, 9, 13, 12,
                10, 11, 6, 7, 4, 5, 18, 16, 17, 19};
        default:
            throw std::runtime_error(
                "perm_vtk: unsupported tetrahedron degree.");
        }
    }

    std::vector<std::uint16_t> vtk_quadrilateral(int num_nodes)
    {
        switch (num_nodes) {
        case 4:
            return {0, 1, 3, 2};
        case 9:
            return {0, 1, 3, 2, 4, 6, 7, 5, 8};
        case 16:
            return {0, 1, 3, 2, 4, 5, 8, 9, 10, 11, 6, 7, 12, 13, 14, 15};
        default:
            throw std::runtime_error(
                "perm_vtk: unsupported quadrilateral degree.");
        }
    }

    std::vector<std::uint16_t> vtk_hexahedron(int num_nodes)
    {
        switch (num_nodes) {
        case 8:
            return {0, 1, 3, 2, 4, 5, 7, 6};
        case 20:
            return {0, 1, 3, 2, 4, 5, 7, 6, 8, 11, 13, 9, 16, 18, 19,
                17, 10, 12, 15, 14};
        case 27:
            return {0, 1, 3, 2, 4, 5, 7, 6, 8, 11, 13, 9, 16, 18, 19,
                17, 10, 12, 15, 14, 22, 23, 21, 24, 20, 25, 26};
        default:
            throw std::runtime_error(
                "perm_vtk: unsupported hexahedron degree.");
        }
    }

} // namespace

int io::cells::cell_degree(mesh::CellType type, int num_nodes)
{
    switch (type) {
    case mesh::CellType::point:
        return num_nodes == 1 ? 0 : -1;
    case mesh::CellType::interval:
        return num_nodes - 1;
    case mesh::CellType::triangle:
        for (int d = 0; d <= 5; ++d)
            if ((d + 1) * (d + 2) / 2 == num_nodes)
                return d;
        break;
    case mesh::CellType::tetrahedron:
        for (int d = 0; d <= 5; ++d)
            if ((d + 1) * (d + 2) * (d + 3) / 6 == num_nodes)
                return d;
        break;
    case mesh::CellType::quadrilateral: {
        const int n = static_cast<int>(std::sqrt(num_nodes));
        return n * n == num_nodes ? n - 1 : -1;
    }
    case mesh::CellType::hexahedron: {
        const int n = static_cast<int>(std::cbrt(num_nodes));
        return n * n * n == num_nodes ? n - 1 : -1;
    }
    default:
        break;
    }
    return -1;
}

std::vector<std::uint16_t> io::cells::perm_vtk(mesh::CellType type,
    int num_nodes)
{
    switch (type) {
    case mesh::CellType::point:
        return {0};
    case mesh::CellType::interval:
        return vtk_interval(num_nodes);
    case mesh::CellType::triangle:
        return vtk_triangle(num_nodes);
    case mesh::CellType::tetrahedron:
        return vtk_tetrahedron(num_nodes);
    case mesh::CellType::quadrilateral:
        return vtk_quadrilateral(num_nodes);
    case mesh::CellType::hexahedron:
        return vtk_hexahedron(num_nodes);
    case mesh::CellType::prism:
    case mesh::CellType::pyramid:
        throw std::runtime_error(
            "perm_vtk: prism and pyramid cells are not supported.");
    }
    throw std::runtime_error("perm_vtk: unknown cell type.");
}

std::vector<std::uint16_t> io::cells::transpose(
    std::span<const std::uint16_t> map)
{
    std::vector<std::uint16_t> t(map.size());
    for (std::size_t i = 0; i < map.size(); ++i)
        t[map[i]] = i;
    return t;
}

std::vector<std::int64_t> io::cells::apply_permutation(
    std::span<const std::int64_t> cells, std::array<std::size_t, 2> shape,
    std::span<const std::uint16_t> p)
{
    assert(cells.size() == shape[0] * shape[1]);
    std::vector<std::int64_t> result(cells.size());
    for (std::size_t c = 0; c < shape[0]; ++c)
        for (std::size_t i = 0; i < shape[1]; ++i)
            result[c * shape[1] + i] = cells[c * shape[1] + p[i]];
    return result;
}

std::int8_t io::cells::get_vtk_cell_type(mesh::CellType cell)
{
    switch (cell) {
    case mesh::CellType::point:
        return 1;
    case mesh::CellType::interval:
        return 3;
    case mesh::CellType::triangle:
        return 5;
    case mesh::CellType::quadrilateral:
        return 9;
    case mesh::CellType::tetrahedron:
        return 10;
    case mesh::CellType::hexahedron:
        return 12;
    case mesh::CellType::prism:
        return 13;
    case mesh::CellType::pyramid:
        return 14;
    }
    throw std::runtime_error("get_vtk_cell_type: unknown cell type.");
}

std::tuple<mesh::CellType, std::int8_t> io::cells::vtk_to_dolfinx(
    std::int8_t vtk_type)
{
    using enum mesh::CellType;
    switch (vtk_type) {
    case 1:
        return {point, 0};
    case 3:
        return {interval, 1};
    case 5:
        return {triangle, 1};
    case 9:
        return {quadrilateral, 1};
    case 10:
        return {tetrahedron, 1};
    case 12:
        return {hexahedron, 1};
    case 13:
        return {prism, 1};
    case 14:
        return {pyramid, 1};
    case 21:
        return {interval, 2};
    case 22:
        return {triangle, 2};
    case 23:
        return {quadrilateral, 2};
    case 24:
        return {tetrahedron, 2};
    case 25:
        return {hexahedron, 2};
    case 26:
        return {prism, 2};
    case 27:
        return {pyramid, 2};
    case 68:
        return {interval, -1};
    case 69:
        return {triangle, -1};
    case 70:
        return {quadrilateral, -1};
    case 71:
        return {tetrahedron, -1};
    case 72:
        return {hexahedron, -1};
    case 73:
        return {prism, -1};
    case 74:
        return {pyramid, -1};
    }
    throw std::runtime_error("vtk_to_dolfinx: unknown VTK cell type.");
}
