// hellofem::mesh — topology, entities and connectivity tests
// SPDX-License-Identifier: MIT

#include "mesh/Mesh.h"
#include "mesh/MeshTags.h"
#include "mesh/Topology.h"
#include "mesh/cell_types.h"

#include "basis/element-families.h"
#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "common/IndexMap.h"
#include "fem/CoordinateElement.h"
#include "graph/AdjacencyList.h"

#include <cstdint>
#include <memory>
#include <numeric>
#include <span>
#include <vector>

using namespace hellofem;
using namespace hellofem::mesh;

namespace {

    /// Build a 2x2x2 hexahedral box topology (8 cells, 27 vertices) via
    /// create_topology with the hex vertex ordering.
    std::shared_ptr<Topology> hex_box()
    {
        // 2x2x2 vertex lattice, index = i + j*3 + k*9.
        const auto v = [](int i, int j, int k) -> std::int64_t {
            return i + j * 3 + k * 9;
        };
        // hex: 8 vertices, (0,0,0),(1,0,0),(0,1,0),(1,1,0),
        // (0,0,1),(1,0,1),(0,1,1),(1,1,1).
        const int hex_verts[8][3]
            = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0},
                {0, 0, 1}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}};

        std::vector<std::int64_t> cells;
        std::vector<std::int64_t> orig;
        for (int k = 0; k < 2; ++k)
            for (int j = 0; j < 2; ++j)
                for (int i = 0; i < 2; ++i) {
                    for (int vi = 0; vi < 8; ++vi)
                        cells.push_back(v(i + hex_verts[vi][0],
                            j + hex_verts[vi][1], k + hex_verts[vi][2]));
                    orig.push_back(cells.size() / 8 - 1);
                }

        return std::make_shared<Topology>(create_topology(
            std::span<const std::int64_t>(cells),
            std::span<const std::int64_t>(orig), CellType::hexahedron, 1));
    }

    /// Number of entities of dimension `dim` in a topology.
    std::int32_t num_entities(const Topology& topology, int dim)
    {
        auto map = topology.index_map(dim);
        return map ? map->size_local() + map->num_ghosts() : -1;
    }

} // namespace

TEST_CASE("Hex box topology: entities", "[mesh]")
{
    auto topology = hex_box();
    REQUIRE(topology->dim() == 3);
    REQUIRE(topology->cell_type() == CellType::hexahedron);

    // Vertices.
    REQUIRE(num_entities(*topology, 0) == 27);

    // Cells.
    REQUIRE(num_entities(*topology, 3) == 8);

    // Create edges (dim 1) and facets (dim 2).
    REQUIRE(topology->create_entities(1));
    REQUIRE(topology->create_entities(2));
    REQUIRE(topology->create_entities(3) == false); // already exist

    // 2x2x2 hex box: 27 vertices (3x3x3 lattice).
    // Edges: 3 directions x 2 intervals x (3x3 perpendicular) = 54.
    REQUIRE(num_entities(*topology, 1) == 54);
    // Facets: 3 directions x 3 planes x (2x2 grid) = 36.
    REQUIRE(num_entities(*topology, 2) == 36);
}

TEST_CASE("Hex box topology: connectivity", "[mesh]")
{
    auto topology = hex_box();
    topology->create_entities(1);
    topology->create_entities(2);

    // cell-facet connectivity: 8 cells x 6 facets.
    topology->create_connectivity(3, 2);
    auto c_to_f = topology->connectivity(3, 2);
    REQUIRE(c_to_f != nullptr);
    REQUIRE(c_to_f->num_nodes() == 8);
    for (std::int32_t c = 0; c < 8; ++c)
        REQUIRE(c_to_f->num_links(c) == 6);

    // facet-cell connectivity: each interior facet connects 2 cells.
    topology->create_connectivity(2, 3);
    auto f_to_c = topology->connectivity(2, 3);
    REQUIRE(f_to_c != nullptr);
    REQUIRE(f_to_c->num_nodes() == 36);
    // 12 interior facets have 2 cells, 24 boundary have 1.
    int pairs = 0, singles = 0;
    for (std::int32_t f = 0; f < 36; ++f) {
        int n = f_to_c->num_links(f);
        if (n == 2)
            pairs++;
        else if (n == 1)
            singles++;
    }
    REQUIRE(pairs == 12);
    REQUIRE(singles == 24);
}

TEST_CASE("Hex box topology: permutations", "[mesh]")
{
    auto topology = hex_box();
    topology->create_entity_permutations();

    const auto& cell_info = topology->get_cell_permutation_info();
    REQUIRE(cell_info.size() == 8);
    const auto& facet_perm = topology->get_facet_permutations();
    REQUIRE(facet_perm.size() == 8 * 6);
}

TEST_CASE("Hex box topology: mixed cell pairs", "[mesh]")
{
    auto topology = hex_box();
    topology->create_entities(2);
    topology->create_connectivity(3, 2);
    topology->create_connectivity(2, 3);

    // Single cell type: one (0,0) entry with interior facet pairs.
    auto pairs = compute_mixed_cell_pairs(*topology, CellType::quadrilateral);
    REQUIRE(pairs.size() == 1);
    // 12 interior facets * 4 ints per pair.
    REQUIRE(pairs[0].size() == 12 * 4);
}

TEST_CASE("create_meshtags on hex box", "[mesh]")
{
    auto topology = hex_box();
    topology->create_entities(2);
    topology->create_connectivity(2, 0);
    auto f_to_v = topology->connectivity(2, 0);
    REQUIRE(f_to_v != nullptr);

    // Identify the x=0 facets: all their vertices have x coordinate 0.
    std::vector<std::int32_t> tagged;
    for (std::int32_t f = 0; f < 36; ++f) {
        auto verts = f_to_v->links(f);
        bool on_x0 = true;
        for (auto v : verts) {
            if ((v % 3) != 0) {
                on_x0 = false;
                break;
            }
        }
        if (on_x0)
            tagged.push_back(f);
    }
    REQUIRE(tagged.size() == 4);

    // Build a MeshTags from the tagged facets (defined by their
    // vertices), with value 1. Build a variable-length adjacency list
    // with one row per tagged facet.
    std::vector<std::int32_t> facet_verts;
    std::vector<std::int32_t> offsets {0};
    for (auto f : tagged) {
        auto verts = f_to_v->links(f);
        facet_verts.insert(facet_verts.end(), verts.begin(), verts.end());
        offsets.push_back(facet_verts.size());
    }
    graph::AdjacencyList<std::int32_t> entities(facet_verts, offsets);
    std::vector<std::int32_t> values(tagged.size(), 1);
    auto tags = create_meshtags(topology, 2, entities, values, "x0");

    REQUIRE(tags.indices().size() == 4);
    REQUIRE(tags.values().size() == 4);
    REQUIRE(tags.find(1).size() == 4);
    REQUIRE(tags.find(0).empty());
    REQUIRE(tags.dim() == 2);
}

TEST_CASE("create_topology with contiguous vertex indices", "[mesh]")
{
    // A single interval cell with vertices 0, 1.
    std::vector<std::int64_t> cells {0, 1};
    std::vector<std::int64_t> orig {0};
    auto topology = create_topology(std::span(cells), std::span(orig),
        CellType::interval, 1);

    REQUIRE(topology.dim() == 1);
    REQUIRE(num_entities(topology, 0) == 2);
    REQUIRE(num_entities(topology, 1) == 1);
    auto c_to_v = topology.connectivity(1, 0);
    REQUIRE(c_to_v->num_links(0) == 2);
}

TEST_CASE("Mesh with linear geometry on a hex box", "[mesh]")
{
    auto topology = hex_box();

    // Geometry: 27 nodes at a unit lattice, linear Lagrange map.
    std::vector<double> x(27 * 3, 0.0);
    for (int k = 0; k < 3; ++k)
        for (int j = 0; j < 3; ++j)
            for (int i = 0; i < 3; ++i) {
                const int n = i + j * 3 + k * 9;
                x[3 * n + 0] = i;
                x[3 * n + 1] = j;
                x[3 * n + 2] = k;
            }
    std::vector<std::int64_t> nodes(27);
    std::iota(nodes.begin(), nodes.end(), 0);

    // xdofs: per-cell geometry dofmap using global node indices
    // (8 nodes per hex cell).
    auto c_to_v = topology->connectivity(3, 0);
    std::vector<std::int64_t> xdofs;
    for (std::int32_t c = 0; c < 8; ++c)
        for (auto v : c_to_v->links(c))
            xdofs.push_back(v);
    REQUIRE(xdofs.size() == 64);

    fem::CoordinateElement<double> element(
        CellType::hexahedron, 1, basis::element::lagrange_variant::equispaced);
    auto geometry = create_geometry(*topology,
        std::vector<fem::CoordinateElement<double>> {element},
        std::span<const std::int64_t>(nodes),
        std::span<const std::int64_t>(xdofs), x, 3);

    REQUIRE(geometry.dim() == 3);
    REQUIRE(geometry.x().size() == 27 * 3);
    // The geometry dofmap should map each cell to its 8 corner nodes.
    REQUIRE(geometry.cmaps().size() == 1);
    REQUIRE(geometry.cmaps()[0].dim() == 8);

    auto dofmaps = geometry.dofmaps();
    REQUIRE(dofmaps.size() == 1);
    REQUIRE(dofmaps[0].extent(0) == 8); // cells
    REQUIRE(dofmaps[0].extent(1) == 8); // dofs per cell

    // Verify the first cell's dofs give corner coordinates of cell (0,0,0).
    // hex vertex ordering maps local vertex i to (i, j, k) lattice.
    const int hex_verts[8][3]
        = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0},
            {0, 0, 1}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}};
    for (int vi = 0; vi < 8; ++vi) {
        const std::int32_t dof = dofmaps[0](0, vi);
        const auto& X = geometry.x();
        REQUIRE(X[3 * dof + 0] == Catch::Approx(hex_verts[vi][0]));
        REQUIRE(X[3 * dof + 1] == Catch::Approx(hex_verts[vi][1]));
        REQUIRE(X[3 * dof + 2] == Catch::Approx(hex_verts[vi][2]));
    }

    // Build the Mesh object.
    Mesh<double> mesh(std::move(topology), std::move(geometry));
    REQUIRE(mesh.topology()->dim() == 3);
    REQUIRE(mesh.geometry().x().size() == 27 * 3);
}
