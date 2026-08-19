// hellofem::io — COMSOL .mphtxt mesh reader
// SPDX-License-Identifier: MIT

#include "mphtxt.h"

#include "basis/element-families.h"
#include "fem/CoordinateElement.h"
#include "mesh/Geometry.h"
#include "mesh/Topology.h"
#include "mesh/cell_types.h"
#include "mesh/utils.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <numeric>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace hellofem;

namespace {

    /// A parsed element block of a .mphtxt file.
    struct Block {
        mesh::CellType type; ///< Cell shape.
        int order; ///< 1 or 2 (Lagrange).
        std::vector<std::int64_t> elements; ///< Node indices (flat).
        std::vector<int> geom_indices; ///< Geometric entity index per element.
    };

    /// Parsed content of a .mphtxt file.
    struct Parsed {
        int sdim = 3; ///< Spatial dimension.
        int lowest_vertex_index = 0; ///< Index of the first mesh vertex.
        std::vector<std::array<double, 3>> vertices; ///< Coordinates.
        std::vector<Block> blocks; ///< Element blocks.
    };

    /// Map a .mphtxt element type name to a cell shape, order and
    /// topological dimension.
    std::tuple<mesh::CellType, int, int>
    parse_type(const std::string& name)
    {
        using mesh::CellType;
        const bool order2 = name.size() > 3
            and name.compare(name.size() - 1, 1, "2") == 0;
        const std::string base
            = order2 ? name.substr(0, name.size() - 1) : name;
        if (base == "vtx")
            return {CellType::point, order2 ? 2 : 1, 0};
        if (base == "edg" or base == "lin")
            return {CellType::interval, order2 ? 2 : 1, 1};
        if (base == "tri")
            return {CellType::triangle, order2 ? 2 : 1, 2};
        if (base == "quad")
            return {CellType::quadrilateral, order2 ? 2 : 1, 2};
        if (base == "tet")
            return {CellType::tetrahedron, order2 ? 2 : 1, 3};
        if (base == "hex")
            return {CellType::hexahedron, order2 ? 2 : 1, 3};
        if (base == "prism" or base == "wedge" or base == "pyr"
            or base == "pyramid")
            throw std::runtime_error(
                "read_mphtxt: prism and pyramid cells are not supported.");
        throw std::runtime_error("read_mphtxt: unknown element type '" + name
            + "'.");
    }

    /// Node permutation from the COMSOL second-order ordering to the
    /// basix ordering, or an empty map when the orderings coincide.
    ///
    /// COMSOL lists the second-order nodes as the vertices followed by
    /// the edge midpoints in the order `(0,1) (0,2) (1,2) (0,3) (1,3)
    /// (2,3)` for a tetrahedron and `(0,1) (0,2) (1,2)` for a triangle.
    /// basix orders the edges `(2,3) (1,3) (1,2) (0,3) (0,2) (0,1)` for a
    /// tetrahedron and `(1,2) (0,2) (0,1)` for a triangle (matching the
    /// reference-cell topology). The permutation maps a basix position to
    /// the COMSOL source position.
    std::vector<std::uint16_t> comsol_to_basix(mesh::CellType type, int order)
    {
        if (order != 2)
            return {};
        switch (type) {
        case mesh::CellType::tetrahedron:
            // basix edge order: (2,3)(1,3)(1,2)(0,3)(0,2)(0,1)  [indices 4..9]
            // comsol edge order: (0,1)(0,2)(1,2)(0,3)(1,3)(2,3)  [indices 4..9]
            return {0, 1, 2, 3, 9, 8, 6, 7, 5, 4};
        case mesh::CellType::triangle:
            // basix edge order: (1,2)(0,2)(0,1)  [indices 3..5]
            // comsol edge order: (0,1)(0,2)(1,2)  [indices 3..5]
            return {0, 1, 2, 5, 4, 3};
        case mesh::CellType::interval:
            return {};
        default:
            return {};
        }
    }

    /// Read one integer token from a stream, skipping blank and comment
    /// lines.
    std::int64_t read_int(std::istream& in, std::string& line)
    {
        std::int64_t value;
        while (std::getline(in, line)) {
            std::istringstream iss(line);
            if (iss >> value)
                return value;
        }
        throw std::runtime_error("read_mphtxt: unexpected end of file.");
    }

    /// Read `count` integers (from the current line onwards), skipping
    /// blank and comment lines.
    void read_ints(std::istream& in, std::string& line, std::size_t count,
        std::vector<std::int64_t>& out)
    {
        out.reserve(out.size() + count);
        while (out.size() < count) {
            std::istringstream iss(line);
            std::int64_t v;
            while (iss >> v)
                out.push_back(v);
            if (out.size() < count) {
                if (!std::getline(in, line))
                    throw std::runtime_error(
                        "read_mphtxt: unexpected end of file.");
                if (!line.empty() and line[0] == '#')
                    continue;
            }
        }
    }

    Parsed parse(const std::filesystem::path& filename)
    {
        std::ifstream file(filename);
        if (!file)
            throw std::runtime_error("read_mphtxt: cannot open file '"
                + filename.string() + "'.");

        Parsed data;
        std::string line;
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            std::int64_t n;
            if (line.find("# sdim") != std::string::npos) {
                iss >> data.sdim;
            }
            else if (line.find("# lowest mesh vertex index") != std::string::npos) {
                iss >> data.lowest_vertex_index;
            }
            else if (line.find("# number of mesh vertices") != std::string::npos) {
                iss >> n;
                while (std::getline(file, line)
                    and line.find("# Mesh vertex coordinates")
                        == std::string::npos) {
                }
                data.vertices.reserve(n);
                std::size_t count = 0;
                while (count < static_cast<std::size_t>(n)
                    and std::getline(file, line)) {
                    std::istringstream ciss(line);
                    std::array<double, 3> c {0, 0, 0};
                    for (int d = 0; d < data.sdim; ++d)
                        ciss >> c[d];
                    data.vertices.push_back(c);
                    ++count;
                }
            }
            else if (line.find("# Type #") != std::string::npos) {
                // Type name line: "<len> <name> # type name".
                std::string token;
                while (std::getline(file, line)) {
                    std::istringstream tiss(line);
                    if (tiss >> token and tiss >> token)
                        break;
                }
                auto [type, order, dim] = parse_type(token);

                const std::size_t nve
                    = static_cast<std::size_t>(read_int(file, line));
                const std::size_t ne = static_cast<std::size_t>(read_int(file, line));
                while (std::getline(file, line)
                    and line.find("# Elements") == std::string::npos) {
                }

                Block block {type, order, {}, {}};
                block.elements.reserve(ne * nve);
                read_ints(file, line, ne * nve, block.elements);

                // Geometric entity indices (one per element).
                const std::size_t ng
                    = static_cast<std::size_t>(read_int(file, line));
                while (std::getline(file, line)
                    and line.find("# Geometric entity indices")
                        == std::string::npos) {
                }
                block.geom_indices.reserve(ng);
                std::vector<std::int64_t> gi;
                read_ints(file, line, ng, gi);
                block.geom_indices.assign(gi.begin(), gi.end());

                data.blocks.push_back(std::move(block));
            }
        }

        return data;
    }

} // namespace

io::MphtxtMesh io::read_mphtxt(const std::filesystem::path& filename)
{
    Parsed data = parse(filename);
    if (data.sdim < 1 or data.sdim > 3)
        throw std::runtime_error("read_mphtxt: unsupported spatial dimension.");

    // Volume cells are the blocks of maximal topological dimension.
    int tdim = 0;
    for (const Block& block : data.blocks)
        tdim = std::max(tdim, mesh::cell_dim(block.type));

    const std::int64_t offset = data.lowest_vertex_index;
    std::vector<mesh::CellType> volume_types;
    std::vector<std::int64_t> cells; // Cell-to-vertex (flat, per type).
    std::vector<std::int64_t> orig; // Original cell index (identity).
    std::vector<int> cell_geom_indices; // Domain index per volume element.
    for (const Block& block : data.blocks) {
        if (mesh::cell_dim(block.type) != tdim)
            continue;

        const auto perm = comsol_to_basix(block.type, block.order);
        const std::size_t nv = mesh::cell_num_entities(block.type, 0);
        const std::size_t npe = block.elements.size() / block.geom_indices.size();
        for (std::size_t e = 0; e < block.geom_indices.size(); ++e) {
            for (std::size_t i = 0; i < nv; ++i) {
                const std::int64_t node
                    = block.elements[e * npe
                          + (perm.empty() ? i : perm[i])]
                    - offset;
                cells.push_back(node);
            }
            orig.push_back(static_cast<std::int64_t>(orig.size()));
        }
        cell_geom_indices.insert(cell_geom_indices.end(),
            block.geom_indices.begin(), block.geom_indices.end());
        volume_types.push_back(block.type);
    }

    if (volume_types.empty())
        throw std::runtime_error(
            "read_mphtxt: no volume cells found in the file.");
    const mesh::CellType cell_type = volume_types.front();
    if (std::any_of(volume_types.begin(), volume_types.end(),
            [cell_type](mesh::CellType t) { return t != cell_type; }))
        throw std::runtime_error(
            "read_mphtxt: mixed volume cell types are not supported.");
    auto topology = std::make_shared<mesh::Topology>(
        mesh::create_topology(std::span<const std::int64_t>(cells),
            std::span<const std::int64_t>(orig), cell_type, 1));

    // Isoparametric geometry: first order uses the vertex coordinates,
    // second order the full node set (vertices + edge midpoints).
    // Geometry order comes from the volume block.
    int geom_order = 1;
    for (const Block& block : data.blocks)
        if (mesh::cell_dim(block.type) == tdim) {
            geom_order = block.order;
            break;
        }

    const std::size_t num_nodes
        = data.vertices.size();
    std::vector<std::int64_t> nodes(num_nodes);
    std::iota(nodes.begin(), nodes.end(), 0);
    std::vector<double> x(3 * num_nodes, 0.0);
    for (std::size_t i = 0; i < num_nodes; ++i)
        for (int d = 0; d < 3; ++d)
            x[3 * i + d] = data.vertices[i][d];

    std::vector<std::int64_t> xdofs;
    if (geom_order == 1) {
        xdofs = cells; // Vertex connectivity.
    }
    else {
        // Full node connectivity in basix order (for the P2 map).
        for (const Block& block : data.blocks) {
            if (mesh::cell_dim(block.type) != tdim)
                continue;
            const auto perm = comsol_to_basix(block.type, geom_order);
            const std::size_t npe
                = block.elements.size() / block.geom_indices.size();
            for (std::size_t e = 0; e < block.geom_indices.size(); ++e)
                for (std::size_t i = 0; i < npe; ++i)
                    xdofs.push_back(
                        block.elements[e * npe + (perm.empty() ? i : perm[i])]
                        - offset);
        }
    }

    fem::CoordinateElement<double> coord_el(cell_type, geom_order);
    if (geom_order > 1) {
        // The P2 geometry dof layout carries edge (and face) dofs, so
        // the mesh entities must exist before building the dofmap.
        for (int d = 1; d < tdim; ++d)
            topology->create_entities(d);
        if (coord_el.needs_dof_permutations())
            topology->create_entity_permutations(1);
    }
    auto geometry = mesh::create_geometry(*topology,
        std::vector<fem::CoordinateElement<double>> {coord_el},
        std::span<const std::int64_t>(nodes),
        std::span<const std::int64_t>(xdofs), x, data.sdim);

    // Boundary facet tags from the co-dimension-1 blocks.
    // Note: for geom_order > 1, topological entities (dim 1..tdim-1)
    // are already created above; avoid re-creating dim 2 which triggers a crash.
    if (geom_order == 1)
        topology->create_entities(tdim - 1);

    // Collect all boundary entity vertex lists (flat) and their geometric
    // entity indices; map them to facet indices in one pass.
    std::vector<std::int32_t> entity_vertices;
    std::vector<int> entity_geom_indices;
    for (const Block& block : data.blocks) {
        if (mesh::cell_dim(block.type) != tdim - 1)
            continue;
        const auto perm = comsol_to_basix(block.type, block.order);
        const std::size_t nv = mesh::cell_num_entities(block.type, 0);
        const std::size_t npe
            = block.elements.size() / block.geom_indices.size();
        for (std::size_t e = 0; e < block.geom_indices.size(); ++e) {
            for (std::size_t i = 0; i < nv; ++i) {
                const std::int64_t node
                    = block.elements[e * npe + (perm.empty() ? i : perm[i])]
                    - offset;
                entity_vertices.push_back(static_cast<std::int32_t>(node));
            }
            entity_geom_indices.push_back(block.geom_indices[e]);
        }
    }

    std::vector<std::int32_t> facet_indices;
    std::vector<int> facet_values;
    if (!entity_vertices.empty()) {
        const auto facet_map
            = mesh::entities_to_index(*topology, tdim - 1, entity_vertices);
        for (std::size_t i = 0; i < facet_map.size(); ++i) {
            if (facet_map[i] >= 0) {
                facet_indices.push_back(facet_map[i]);
                facet_values.push_back(entity_geom_indices[i]);
            }
        }
    }

    std::shared_ptr<mesh::MeshTags<int>> facet_tags;
    if (!facet_indices.empty()) {
        // MeshTags requires sorted, unique indices.
        std::vector<std::size_t> order(facet_indices.size());
        std::iota(order.begin(), order.end(), 0);
        std::ranges::sort(order, [&](std::size_t a, std::size_t b) {
            return facet_indices[a] < facet_indices[b];
        });
        std::vector<std::int32_t> sorted_indices;
        std::vector<int> sorted_values;
        sorted_indices.reserve(order.size());
        sorted_values.reserve(order.size());
        for (std::size_t i = 0; i < order.size(); ++i) {
            if (i == 0
                or facet_indices[order[i]] != facet_indices[order[i - 1]]) {
                sorted_indices.push_back(facet_indices[order[i]]);
                sorted_values.push_back(facet_values[order[i]]);
            }
        }
        facet_tags = std::make_shared<mesh::MeshTags<int>>(topology, tdim - 1,
            std::move(sorted_indices), std::move(sorted_values), "facet_tags");
    }

    std::shared_ptr<mesh::MeshTags<int>> cell_tags;
    if (!cell_geom_indices.empty()) {
        const std::size_t nc = cell_geom_indices.size();
        std::vector<std::int32_t> cell_indices(nc);
        std::iota(cell_indices.begin(), cell_indices.end(), 0);
        // Every cell carries a domain index; indices are already sorted/unique.
        cell_tags = std::make_shared<mesh::MeshTags<int>>(topology, tdim,
            std::move(cell_indices), std::move(cell_geom_indices), "cell_tags");
    }

    return MphtxtMesh {
        mesh::Mesh<double>(topology, std::move(geometry)),
        geom_order,
        std::move(facet_tags), std::move(cell_tags)};
}
