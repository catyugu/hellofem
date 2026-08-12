// hellofem::io — cell ordering tests
// SPDX-License-Identifier: MIT

#include "catch2/catch_test_macros.hpp"
#include "io/cells.h"
#include "mesh/cell_types.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <span>
#include <vector>

using namespace hellofem;

namespace {

    /// Apply a permutation to a sequence: `a_p[i] = a[p[i]]`.
    std::vector<std::int64_t> permute(std::span<const std::int64_t> a,
        std::span<const std::uint16_t> p)
    {
        std::vector<std::int64_t> r(p.size());
        for (std::size_t i = 0; i < p.size(); ++i)
            r[i] = a[p[i]];
        return r;
    }

} // namespace

TEST_CASE("cell_degree recovers the Lagrange degree", "[io][cells]")
{
    using mesh::CellType;
    REQUIRE(io::cells::cell_degree(CellType::interval, 2) == 1);
    REQUIRE(io::cells::cell_degree(CellType::interval, 3) == 2);
    REQUIRE(io::cells::cell_degree(CellType::triangle, 3) == 1);
    REQUIRE(io::cells::cell_degree(CellType::triangle, 6) == 2);
    REQUIRE(io::cells::cell_degree(CellType::triangle, 10) == 3);
    REQUIRE(io::cells::cell_degree(CellType::tetrahedron, 4) == 1);
    REQUIRE(io::cells::cell_degree(CellType::tetrahedron, 10) == 2);
    REQUIRE(io::cells::cell_degree(CellType::tetrahedron, 20) == 3);
    REQUIRE(io::cells::cell_degree(CellType::quadrilateral, 4) == 1);
    REQUIRE(io::cells::cell_degree(CellType::quadrilateral, 9) == 2);
    REQUIRE(io::cells::cell_degree(CellType::hexahedron, 8) == 1);
    REQUIRE(io::cells::cell_degree(CellType::hexahedron, 27) == 2);
    REQUIRE(io::cells::cell_degree(CellType::hexahedron, 999) == -1);
}

TEST_CASE("perm_vtk maps are valid permutations", "[io][cells]")
{
    using mesh::CellType;
    const std::vector<std::pair<CellType, int>> cases {
        {CellType::interval, 4}, {CellType::triangle, 3},
        {CellType::triangle, 6}, {CellType::triangle, 10},
        {CellType::tetrahedron, 4}, {CellType::tetrahedron, 10},
        {CellType::tetrahedron, 20}, {CellType::quadrilateral, 16},
        {CellType::hexahedron, 27}};
    for (auto [type, nodes] : cases) {
        auto p = io::cells::perm_vtk(type, nodes);
        REQUIRE(p.size() == static_cast<std::size_t>(nodes));
        auto sorted = p;
        std::ranges::sort(sorted);
        REQUIRE(std::ranges::unique(sorted).begin() == sorted.end());
    }
}

TEST_CASE("perm_vtk and transpose are mutually inverse", "[io][cells]")
{
    using mesh::CellType;
    const std::vector<std::pair<CellType, int>> cases {
        {CellType::triangle, 10}, {CellType::tetrahedron, 20},
        {CellType::quadrilateral, 9}, {CellType::hexahedron, 27},
        {CellType::interval, 4}, {CellType::triangle, 6}};
    for (auto [type, nodes] : cases) {
        auto p = io::cells::perm_vtk(type, nodes);
        auto t = io::cells::transpose(p);
        // permute(permute(x, p), t) == x for any x.
        std::vector<std::int64_t> x(nodes);
        for (int i = 0; i < nodes; ++i)
            x[i] = 100 + i;
        auto y = permute(x, p);
        auto z = permute(y, t);
        REQUIRE(z == x);
    }
}

TEST_CASE("known VTK vertex permutations", "[io][cells]")
{
    using mesh::CellType;
    // Quadrilateral and hexahedron vertices are re-ordered in VTK.
    REQUIRE(io::cells::perm_vtk(CellType::quadrilateral, 4)
        == std::vector<std::uint16_t>({0, 1, 3, 2}));
    REQUIRE(io::cells::perm_vtk(CellType::hexahedron, 8)
        == std::vector<std::uint16_t>({0, 1, 3, 2, 4, 5, 7, 6}));
    // Simplices keep their vertex order.
    REQUIRE(io::cells::perm_vtk(CellType::triangle, 3)
        == std::vector<std::uint16_t>({0, 1, 2}));
    REQUIRE(io::cells::perm_vtk(CellType::tetrahedron, 4)
        == std::vector<std::uint16_t>({0, 1, 2, 3}));
}

TEST_CASE("apply_permutation permutes each cell", "[io][cells]")
{
    using mesh::CellType;
    // Two hexahedra with consecutive node ids.
    std::vector<std::int64_t> cells {0, 1, 2, 3, 4, 5, 6, 7,
        10, 11, 12, 13, 14, 15, 16, 17};
    auto p = io::cells::perm_vtk(CellType::hexahedron, 8);
    auto r = io::cells::apply_permutation(cells, {2, 8}, p);
    REQUIRE(r == std::vector<std::int64_t>({0, 1, 3, 2, 4, 5, 7, 6, 10, 11, 13, 12, 14, 15, 17, 16}));
}

TEST_CASE("VTK cell type round trip", "[io][cells]")
{
    using mesh::CellType;
    for (auto cell : {CellType::interval, CellType::triangle,
             CellType::quadrilateral, CellType::tetrahedron,
             CellType::hexahedron}) {
        const std::int8_t vtk = io::cells::get_vtk_cell_type(cell);
        auto [back, degree] = io::cells::vtk_to_dolfinx(vtk);
        REQUIRE(back == cell);
        REQUIRE(degree == 1);
    }
}
