// hellofem::mesh — mesh utility tests
// SPDX-License-Identifier: MIT

#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "mesh/Geometry.h"
#include "mesh/Mesh.h"
#include "mesh/Topology.h"
#include "mesh/cell_types.h"
#include "mesh/generation.h"
#include "mesh/utils.h"

#include <array>
#include <cstdint>
#include <numeric>
#include <span>
#include <vector>

using namespace hellofem;
using namespace hellofem::mesh;
using Catch::Approx;

namespace {

    /// Number of entities of dimension `dim` in a topology.
    std::int32_t num_entities(const Topology& topology, int dim)
    {
        auto map = topology.index_map(dim);
        return map ? map->size_local() + map->num_ghosts() : -1;
    }

    /// Coordinates of vertex `v` (as an array) via compute_vertex_coords.
    std::array<double, 3> vertex_coord(
        const std::vector<double>& x, std::size_t num_vertices, int v)
    {
        return {x[0 * num_vertices + v], x[1 * num_vertices + v],
            x[2 * num_vertices + v]};
    }

} // namespace

TEST_CASE("exterior_facet_indices on the unit square", "[mesh][utils]")
{
    const int n = 4;
    auto mesh = create_unit_square(n);
    auto topology = mesh->topology_mutable();
    topology->create_entities(1);
    topology->create_connectivity(1, 2);

    // A triangle mesh on an n x n square has 4n boundary edges.
    auto facets = exterior_facet_indices(*topology);
    REQUIRE(facets.size() == 4 * n);

    // Each exterior facet is attached to exactly one cell.
    auto f_to_c = topology->connectivity(1, 2);
    for (auto f : facets)
        REQUIRE(f_to_c->num_links(f) == 1);
}

TEST_CASE("locate_entities marks vertices on a plane", "[mesh][utils]")
{
    const int n = 4;
    auto mesh = create_unit_square(n);

    // Mark the left edge (x == 0).
    auto marker = [](md::mdspan<const double,
                      md::extents<std::size_t, 3, md::dynamic_extent>>
                          x) {
        std::vector<std::int8_t> marked(x.extent(1), 0);
        for (std::size_t i = 0; i < x.extent(1); ++i)
            marked[i] = x(0, i) < 1e-12;
        return marked;
    };

    // n+1 vertices lie on the left edge; each lies on a boundary facet.
    auto vertices = locate_entities(*mesh, 0, marker);
    REQUIRE(vertices.size() == n + 1);

    // n boundary edges lie on the left edge.
    auto edges = locate_entities_boundary(*mesh, 1, marker);
    REQUIRE(edges.size() == n);

    // locate_entities over the whole mesh also finds the left edges.
    auto edges_all = locate_entities(*mesh, 1, marker);
    REQUIRE(edges_all.size() == n);
}

TEST_CASE("compute_vertex_coords and compute_midpoints", "[mesh][utils]")
{
    const int n = 3;
    auto mesh = create_unit_square(n);

    const auto [x, shape] = compute_vertex_coords(*mesh);
    REQUIRE(shape == std::array<std::size_t, 2> {3, 16});
    REQUIRE(vertex_coord(x, 16, 0)[0] == Approx(0.0)); // vertex 0 = (0, 0)
    REQUIRE(vertex_coord(x, 16, 4)[1] == Approx(1.0 / 3)); // vertex 4, y = 1/3

    // Cell midpoints equal the average of the cell vertex coordinates.
    auto topology = mesh->topology_mutable();
    topology->create_entities(2);
    const std::int32_t num_cells = num_entities(*topology, 2);
    std::vector<std::int32_t> cells(num_cells);
    std::iota(cells.begin(), cells.end(), 0);
    auto mid = compute_midpoints(*mesh, 2, cells);

    auto c_to_v = topology->connectivity(2, 0);
    for (std::int32_t c = 0; c < num_cells; ++c) {
        auto verts = c_to_v->links(c);
        double mx = 0, my = 0;
        for (auto v : verts) {
            auto p = vertex_coord(x, 16, v);
            mx += p[0];
            my += p[1];
        }
        REQUIRE(mid[3 * c + 0] == Approx(mx / verts.size()));
        REQUIRE(mid[3 * c + 1] == Approx(my / verts.size()));
        REQUIRE(mid[3 * c + 2] == Approx(0.0));
    }
}

TEST_CASE("entities_to_geometry locates cell vertex coordinates",
    "[mesh][utils]")
{
    const int n = 2;
    auto mesh = create_unit_square(n);

    auto topology = mesh->topology_mutable();
    topology->create_entities(2);
    const std::int32_t num_cells = num_entities(*topology, 2);
    std::vector<std::int32_t> cells(num_cells);
    std::iota(cells.begin(), cells.end(), 0);

    // For each cell the geometry dofs of its closure are its vertices;
    // looking up the coordinates at those geometry nodes reproduces the
    // analytic vertex coordinates.
    const auto [e_to_g, eshape] = entities_to_geometry(*mesh, 2, cells);
    REQUIRE(eshape
        == std::array<std::size_t, 2> {
            static_cast<std::size_t>(num_cells), 3});

    const auto [x, xshape] = compute_vertex_coords(*mesh);
    auto c_to_v = topology->connectivity(2, 0);
    std::span<const double> xg = mesh->geometry().x();
    for (std::int32_t c = 0; c < num_cells; ++c) {
        auto verts = c_to_v->links(c);
        for (std::size_t j = 0; j < 3; ++j) {
            const std::int32_t node = e_to_g[c * 3 + j];
            auto p = vertex_coord(x, xshape[1], verts[j]);
            REQUIRE(xg[3 * node + 0] == Approx(p[0]));
            REQUIRE(xg[3 * node + 1] == Approx(p[1]));
        }
    }
}

TEST_CASE("create_mesh round-trips a triangle mesh", "[mesh][utils]")
{
    // Two triangles forming the unit square (basix triangle ordering).
    std::vector<std::int64_t> cells {0, 1, 2, 1, 3, 2};
    std::vector<double> x {0, 0, 1, 0, 0, 1, 1, 1};
    auto mesh = create_mesh(std::span<const std::int64_t>(cells),
        CellType::triangle, x, 2);

    auto topology = mesh.topology();
    REQUIRE(topology->dim() == 2);
    REQUIRE(topology->cell_type() == CellType::triangle);
    REQUIRE(num_entities(*topology, 0) == 4);
    REQUIRE(num_entities(*topology, 2) == 2);

    // Geometry nodes map 1:1 to the input vertices.
    const auto [xc, shape] = compute_vertex_coords(mesh);
    REQUIRE(shape == std::array<std::size_t, 2> {3, 4});
    REQUIRE(vertex_coord(xc, 4, 2)[0] == Approx(0.0)); // vertex 2: (0, 1)
    REQUIRE(vertex_coord(xc, 4, 2)[1] == Approx(1.0));
}

TEST_CASE("create_rectangle and create_box generate expected meshes",
    "[mesh][utils]")
{
    auto rect = create_rectangle({-1.0, -1.0}, {1.0, 1.0}, {3, 4});
    auto ttopo = rect->topology_mutable();
    REQUIRE(ttopo->dim() == 2);
    REQUIRE(num_entities(*ttopo, 2) == 2 * 3 * 4);
    REQUIRE(num_entities(*ttopo, 0) == 4 * 5);

    auto box = create_box({0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}, {2, 2, 2});
    auto btopo = box->topology_mutable();
    REQUIRE(btopo->dim() == 3);
    REQUIRE(btopo->cell_type() == CellType::hexahedron);
    REQUIRE(num_entities(*btopo, 3) == 8);
    REQUIRE(num_entities(*btopo, 0) == 27);

    // Cell midpoints equal the analytic centroids of the box grid: cell
    // c = i + 2 j + 4 k has centroid (0.25+0.5i, 0.25+0.5j, 0.25+0.5k).
    btopo->create_entities(3);
    std::vector<std::int32_t> cells(8);
    std::iota(cells.begin(), cells.end(), 0);
    auto mid = compute_midpoints(*box, 3, cells);
    for (std::int32_t c = 0; c < 8; ++c) {
        const int i = c % 2, j = (c / 2) % 2, k = c / 4;
        REQUIRE(mid[3 * c + 0] == Approx(0.25 + 0.5 * i));
        REQUIRE(mid[3 * c + 1] == Approx(0.25 + 0.5 * j));
        REQUIRE(mid[3 * c + 2] == Approx(0.25 + 0.5 * k));
    }
}
