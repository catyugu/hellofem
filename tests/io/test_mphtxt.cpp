// hellofem::io — COMSOL .mphtxt mesh reader tests
// SPDX-License-Identifier: MIT

#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "io/mphtxt.h"
#include "mesh/Mesh.h"
#include "mesh/MeshTags.h"
#include "mesh/Topology.h"
#include "mesh/cell_types.h"
#include "mesh/utils.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <vector>

using namespace hellofem;
using Catch::Approx;

namespace {

    /// Write `content` to a temporary file and return its path.
    std::filesystem::path write_temp(const std::string& content)
    {
        const auto path = std::filesystem::temp_directory_path()
            / "hellofem_fixture.mphtxt";
        std::ofstream out(path);
        out << content;
        return path;
    }

    /// Number of entities of dimension `dim`.
    std::int32_t num_entities(const mesh::Topology& topology, int dim)
    {
        auto map = topology.index_map(dim);
        return map ? map->size_local() + map->num_ghosts() : -1;
    }

    /// Sum of the signed volumes of all tetrahedra (basix vertex order),
    /// using the per-vertex coordinates from compute_vertex_coords (the
    /// geometry array is indexed by geometry dofs, which differ from
    /// corner vertex indices for higher-order geometry).
    double total_signed_volume(const mesh::Mesh<double>& mesh)
    {
        auto topology = mesh.topology();
        auto c_to_v = topology->connectivity(3, 0);
        const auto [vc, shape] = mesh::compute_vertex_coords(mesh);
        const std::size_t nv = shape[1];
        auto coord = [&](std::int32_t v) -> std::array<double, 3> {
            return {vc[0 * nv + v], vc[1 * nv + v], vc[2 * nv + v]};
        };
        double total = 0;
        for (std::int32_t c = 0; c < c_to_v->num_nodes(); ++c) {
            auto v = c_to_v->links(c);
            auto p0 = coord(v[0]), p1 = coord(v[1]);
            auto p2 = coord(v[2]), p3 = coord(v[3]);
            double a[3] {p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]};
            double b[3] {p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]};
            double c_[3] {p3[0] - p0[0], p3[1] - p0[1], p3[2] - p0[2]};
            total += a[0] * (b[1] * c_[2] - b[2] * c_[1])
                - a[1] * (b[0] * c_[2] - b[2] * c_[0])
                + a[2] * (b[0] * c_[1] - b[1] * c_[0]);
        }
        return total / 6.0;
    }

    const std::string fixture = R"(# Created by COMSOL Multiphysics.

# Major & minor version
0 1
2 # number of tags
# Tags
5 mesh1
10 mesh1_sel1
2 # number of types
# Types
3 obj
3 obj

# --------- Object 0 ----------

0 0 1
4 Mesh # class
4 # version
3 # sdim
5 # number of mesh vertices
0 # lowest mesh vertex index

# Mesh vertex coordinates
0 0 0
1 0 0
0 1 0
0 0 1
1 0 1

2 # number of element types

# Type #0

3 tet # type name

4 # number of vertices per element
2 # number of elements
# Elements
0 1 2 3
1 4 2 3

2 # number of geometric entity indices
# Geometric entity indices
1
1

# Type #1

3 tri # type name

3 # number of vertices per element
2 # number of elements
# Elements
0 1 2
0 2 3

2 # number of geometric entity indices
# Geometric entity indices
4
8
)";

} // namespace

TEST_CASE("read_mphtxt imports a two-tet fixture", "[io][mphtxt]")
{
    const auto path = write_temp(fixture);
    auto data = io::read_mphtxt(path);

    auto topology = data.mesh.topology();
    REQUIRE(topology->dim() == 3);
    REQUIRE(topology->cell_type() == mesh::CellType::tetrahedron);
    REQUIRE(num_entities(*topology, 0) == 5);
    REQUIRE(num_entities(*topology, 3) == 2);

    // Both cells are properly oriented (positive volume).
    REQUIRE(total_signed_volume(data.mesh) == Approx(1.0 / 3.0));

    // Boundary facet tags carry the geometric entity indices.
    REQUIRE(data.facet_tags != nullptr);
    REQUIRE(data.facet_tags->dim() == 2);
    REQUIRE(data.facet_tags->size() == 2);
    const std::vector<std::int32_t> indices(data.facet_tags->indices().begin(),
        data.facet_tags->indices().end());
    const std::vector<int> values(
        data.facet_tags->values().begin(), data.facet_tags->values().end());
    REQUIRE(indices[0] < indices[1]);
    REQUIRE(values[0] == 4);
    REQUIRE(values[1] == 8);

    // Every tagged facet is a boundary facet.
    auto topology_mut = data.mesh.topology_mutable();
    topology_mut->create_connectivity(2, 3);
    auto ext = mesh::exterior_facet_indices(*topology_mut);
    for (auto f : indices)
        REQUIRE(std::ranges::find(ext, f) != ext.end());
}

TEST_CASE("read_mphtxt imports the busbar mesh (real file)", "[io][mphtxt]")
{
    const auto ref = std::filesystem::path(HELLOFEM_SOURCE_DIR)
        / ".cache/ref/mpfem/cases/busbar_steady/mesh.mphtxt";
    if (!std::filesystem::exists(ref)) {
        WARN("reference mesh not available; skipping");
        return;
    }

    auto data = io::read_mphtxt(ref);
    auto topology = data.mesh.topology();
    REQUIRE(topology->cell_type() == mesh::CellType::tetrahedron);
    REQUIRE(num_entities(*topology, 0) == 7340);
    REQUIRE(num_entities(*topology, 3) == 31021);

    // All cells are oriented consistently (positive total volume).
    REQUIRE(total_signed_volume(data.mesh) > 0.0);

    // Boundary tags match the file's boundary triangle count (9138).
    REQUIRE(data.facet_tags != nullptr);
    REQUIRE(data.facet_tags->size() == 9138);
}

TEST_CASE("read_mphtxt imports a second-order mesh", "[io][mphtxt]")
{
    const auto ref = std::filesystem::path(HELLOFEM_SOURCE_DIR)
        / ".cache/ref/mpfem/cases/busbar_steady_order2/mesh.mphtxt";
    if (!std::filesystem::exists(ref)) {
        WARN("reference mesh not available; skipping");
        return;
    }

    auto data = io::read_mphtxt(ref);
    auto topology = data.mesh.topology();
    REQUIRE(topology->cell_type() == mesh::CellType::tetrahedron);
    // Second-order tet: cells carry 4 corner + 6 edge nodes; the mesh
    // geometry has one point per node (49889 in the file).
    REQUIRE(num_entities(*topology, 3) == 31021);
    REQUIRE(data.mesh.geometry().x().size() == 3 * 49889);
    REQUIRE(num_entities(*topology, 0) > 7000);
    REQUIRE(data.facet_tags->size() == 9138);

    // Geometry is isoparametric P2: the cell volume is positive.
    REQUIRE(total_signed_volume(data.mesh) > 0.0);
}
