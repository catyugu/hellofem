// hellofem::geometry — GJK distance, BoundingBoxTree and query tests
// SPDX-License-Identifier: MIT

#include "basis/element-families.h"
#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "fem/CoordinateElement.h"
#include "geometry/BoundingBoxTree.h"
#include "geometry/gjk.h"
#include "geometry/utils.h"
#include "graph/AdjacencyList.h"
#include "mesh/Geometry.h"
#include "mesh/Mesh.h"
#include "mesh/Topology.h"
#include "mesh/cell_types.h"
#include "mesh/generation.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numeric>
#include <span>
#include <vector>

using namespace hellofem;
using Catch::Approx;

namespace {

    /// GJK distance (scalar) between two point sets.
    template <typename R0, typename R1>
    double gjk_distance(const R0& p0, const R1& q0)
    {
        auto d = geometry::compute_distance_gjk<double>(
            std::span<const double>(p0), std::span<const double>(q0));
        return std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    }

    /// Build a 2x2x2 hexahedron mesh (8 cells, 27 vertices) on [0,2]^3
    /// with linear geometry, following test_mesh.cpp::hex_box.
    std::shared_ptr<mesh::Mesh<double>> hex_box_mesh()
    {
        const auto v = [](std::int64_t i, std::int64_t j, std::int64_t k) {
            return i + j * 3 + k * 9;
        };
        const int hex_verts[8][3]
            = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0},
                {0, 0, 1}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}};

        std::vector<std::int64_t> cells, orig;
        for (int k = 0; k < 2; ++k)
            for (int j = 0; j < 2; ++j)
                for (int i = 0; i < 2; ++i) {
                    for (int vi = 0; vi < 8; ++vi)
                        cells.push_back(
                            v(i + hex_verts[vi][0], j + hex_verts[vi][1],
                                k + hex_verts[vi][2]));
                    orig.push_back(cells.size() / 8 - 1);
                }

        auto topology = std::make_shared<mesh::Topology>(mesh::create_topology(
            std::span<const std::int64_t>(cells),
            std::span<const std::int64_t>(orig), mesh::CellType::hexahedron, 1));

        // Linear geometry on the 27 lattice points.
        constexpr int dim = 3;
        std::vector<double> x(27 * dim, 0.0);
        for (int k = 0; k < 3; ++k)
            for (int j = 0; j < 3; ++j)
                for (int i = 0; i < 3; ++i) {
                    const std::int64_t n = v(i, j, k);
                    x[dim * n + 0] = i;
                    x[dim * n + 1] = j;
                    x[dim * n + 2] = k;
                }
        std::vector<std::int64_t> nodes(27);
        std::iota(nodes.begin(), nodes.end(), 0);
        auto c_to_v = topology->connectivity(3, 0);
        std::vector<std::int64_t> xdofs;
        for (std::int32_t c = 0; c < 8; ++c)
            for (auto w : c_to_v->links(c))
                xdofs.push_back(w);

        fem::CoordinateElement<double> coord_el(mesh::CellType::hexahedron, 1,
            basis::element::lagrange_variant::equispaced);
        auto geometry = mesh::create_geometry(*topology,
            std::vector<fem::CoordinateElement<double>> {coord_el},
            std::span<const std::int64_t>(nodes),
            std::span<const std::int64_t>(xdofs), x, dim);
        return std::make_shared<mesh::Mesh<double>>(
            std::move(topology), std::move(geometry));
    }

} // namespace

TEST_CASE("GJK: point-point and point-segment distances", "[geometry]")
{
    const std::array<double, 3> origin {0, 0, 0};
    const std::array<double, 3> p {3, 4, 0};
    REQUIRE(gjk_distance(origin, p) == Approx(5.0));

    // Point (0,0,0) to the segment [(1,0,0), (2,0,0)]: closest point (1,0,0).
    const std::array<double, 6> seg {1, 0, 0, 2, 0, 0};
    REQUIRE(gjk_distance(origin, seg) == Approx(1.0));

    // Point (0,0,0) to the segment [(-1,0,0), (1,0,0)]: origin on it.
    const std::array<double, 6> seg2 {-1, 0, 0, 1, 0, 0};
    REQUIRE(gjk_distance(origin, seg2) == Approx(0.0));
}

TEST_CASE("GJK: point-triangle and triangle-triangle distances", "[geometry]")
{
    // Triangle in the plane x+y+z=1 with vertices (1,0,0),(0,1,0),(0,0,1).
    const std::array<double, 9> tri {1, 0, 0, 0, 1, 0, 0, 0, 1};
    const std::array<double, 3> origin {0, 0, 0};
    // Distance from origin to the plane x+y+z=1 is 1/sqrt(3).
    REQUIRE(gjk_distance(origin, tri) == Approx(1.0 / std::sqrt(3.0)));

    // Triangle containing the origin: vertices (1,0,0),(0,1,0),(-1,0,0).
    const std::array<double, 9> tri2 {1, 0, 0, 0, 1, 0, -1, 0, 0};
    REQUIRE(gjk_distance(origin, tri2) == Approx(0.0));

    // Parallel skew segments: [(0,0,0),(1,0,0)] vs [(0,1,1),(1,1,1)].
    const std::array<double, 6> seg_a {0, 0, 0, 1, 0, 0};
    const std::array<double, 6> seg_b {0, 1, 1, 1, 1, 1};
    REQUIRE(gjk_distance(seg_a, seg_b) == Approx(std::sqrt(2.0)));
}

TEST_CASE("GJK: point-tetrahedron distance", "[geometry]")
{
    // Tetrahedron with vertices (1,0,0),(0,1,0),(0,0,1),(-1,-1,-1).
    const std::array<double, 12> tet {1, 0, 0, 0, 1, 0, 0, 0, 1, -1, -1, -1};
    const std::array<double, 3> origin {0, 0, 0};
    // Origin inside -> distance 0.
    REQUIRE(gjk_distance(origin, tet) == Approx(0.0));

    // External point (2,0,0) to the vertex (1,0,0).
    const std::array<double, 3> p {2, 0, 0};
    REQUIRE(gjk_distance(p, tet) == Approx(1.0));
}

TEST_CASE("BoundingBoxTree: cells of a 2x2x2 hex box", "[geometry]")
{
    auto mesh = hex_box_mesh();
    geometry::BoundingBoxTree<double> tree(*mesh, 3, 0.0);

    // 8 leaves -> 2*8 - 1 = 15 nodes.
    REQUIRE(tree.num_bboxes() == 15);
    REQUIRE(tree.tdim() == 3);

    // Root bbox covers [0,2]^3.
    auto root = tree.get_bbox(14);
    for (int k = 0; k < 3; ++k) {
        REQUIRE(root[k] == Approx(0.0));
        REQUIRE(root[k + 3] == Approx(2.0));
    }

    // A leaf's bbox is [0,1]^3 and its bbox() entries equal the cell index.
    bool found_cell0 = false;
    for (int node = 0; node < 15; ++node) {
        auto b = tree.bbox(node);
        if (b[0] == b[1] and b[0] == 0) {
            found_cell0 = true;
            auto bb = tree.get_bbox(node);
            REQUIRE(bb[0] == Approx(0.0));
            REQUIRE(bb[1] == Approx(0.0));
            REQUIRE(bb[2] == Approx(0.0));
            REQUIRE(bb[3] == Approx(1.0));
            REQUIRE(bb[4] == Approx(1.0));
            REQUIRE(bb[5] == Approx(1.0));
        }
    }
    REQUIRE(found_cell0);
}

TEST_CASE("BoundingBoxTree: point-cloud tree of cell midpoints", "[geometry]")
{
    auto mesh = hex_box_mesh();
    // Midpoints of the 8 cells: (i+0.5, j+0.5, k+0.5).
    std::vector<std::pair<std::array<double, 3>, std::int32_t>> points;
    for (int k = 0; k < 2; ++k)
        for (int j = 0; j < 2; ++j)
            for (int i = 0; i < 2; ++i) {
                const std::int32_t cell = i + j * 2 + k * 4;
                points.push_back({{i + 0.5, j + 0.5, k + 0.5}, cell});
            }
    geometry::BoundingBoxTree<double> tree(points);
    REQUIRE(tree.num_bboxes() == 15);
    REQUIRE(tree.tdim() == 0);

    // Leaves have lower == upper == midpoint.
    bool found_cell5 = false;
    for (int node = 0; node < 15; ++node) {
        auto b = tree.bbox(node);
        if (b[0] == b[1] and b[0] == 5) { // cell 5: (1,0,1)
            found_cell5 = true;
            auto bb = tree.get_bbox(node);
            REQUIRE(bb[0] == Approx(1.5));
            REQUIRE(bb[1] == Approx(0.5));
            REQUIRE(bb[2] == Approx(1.5));
            REQUIRE(bb[3] == Approx(1.5));
            REQUIRE(bb[4] == Approx(0.5));
            REQUIRE(bb[5] == Approx(1.5));
        }
    }
    REQUIRE(found_cell5);
}

TEST_CASE("geometry queries: collisions, closest, distance", "[geometry]")
{
    auto mesh = hex_box_mesh();
    geometry::BoundingBoxTree<double> tree(*mesh, 3, 0.0);

    // Point (0.5,0.5,0.5) collides with cell 0.
    const std::array<double, 3> inside {0.5, 0.5, 0.5};
    auto hits = geometry::compute_collisions(tree, inside);
    REQUIRE(hits.num_nodes() == 1);
    REQUIRE(hits.links(0).size() == 1);
    REQUIRE(hits.links(0)[0] == 0);

    // External point collides with nothing.
    const std::array<double, 3> outside {3, 3, 3};
    auto hits_out = geometry::compute_collisions(tree, outside);
    REQUIRE(hits_out.links(0).size() == 0);

    // Tree-tree collision: cell-0 midpoint point-tree against the cell tree.
    std::vector<std::pair<std::array<double, 3>, std::int32_t>> mp;
    mp.emplace_back(std::array<double, 3> {0.5, 0.5, 0.5}, 0);
    geometry::BoundingBoxTree<double> mtree(mp);
    auto pairs = geometry::compute_collisions(tree, mtree);
    REQUIRE(pairs.size() == 2);
    REQUIRE(pairs[0] == 0);
    REQUIRE(pairs[1] == 0);

    // Closest entity: near the (0,0,0) corner -> cell 0; near (2,2,2) -> cell 7.
    geometry::BoundingBoxTree<double> cell_midpoints(*mesh, 3, 0.0);
    // Rebuild as a point cloud of cell midpoints via create_midpoint_tree.
    auto mid_tree = geometry::create_midpoint_tree(
        *mesh, 3, std::span<const std::int32_t>(std::vector<std::int32_t> {0, 1, 2, 3, 4, 5, 6, 7}));
    const std::array<double, 6> two_points {
        0.1, 0.1, 0.1, 1.9, 1.9, 1.9};
    auto closest = geometry::compute_closest_entity(tree, mid_tree, *mesh, two_points);
    REQUIRE(closest.size() == 2);
    REQUIRE(closest[0] == 0);
    REQUIRE(closest[1] == 7);

    // Squared distance from (1.1, 0.5, 0.5) to cell 0 (bbox [0,1]^3).
    const std::array<double, 3> near {1.1, 0.5, 0.5};
    const std::vector<std::int32_t> cell0 {0};
    auto d2 = geometry::squared_distance(*mesh, 3, cell0, near);
    REQUIRE(d2.size() == 1);
    REQUIRE(d2[0] == Approx(0.01));

    // Colliding cells: point (0.5,0.5,0.5) among all 8 candidate cells.
    graph::AdjacencyList<std::int32_t> candidates(
        std::vector<std::int32_t> {0, 1, 2, 3, 4, 5, 6, 7},
        std::vector<std::int32_t> {0, 8});
    auto colliding = geometry::compute_colliding_cells(*mesh, candidates, inside);
    REQUIRE(colliding.num_nodes() == 1);
    REQUIRE(colliding.links(0).size() == 1);
    REQUIRE(colliding.links(0)[0] == 0);
}
