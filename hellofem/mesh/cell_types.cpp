// hellofem::mesh — cell type identifiers and reference-cell queries
// SPDX-License-Identifier: MIT

#include "cell_types.h"

#include "basis/cell.h"

#include <cassert>
#include <cmath>
#include <format>
#include <stdexcept>
#include <vector>

namespace hellofem::mesh {

    std::string to_string(CellType type)
    {
        switch (type) {
        case CellType::point:
            return "point";
        case CellType::interval:
            return "interval";
        case CellType::triangle:
            return "triangle";
        case CellType::tetrahedron:
            return "tetrahedron";
        case CellType::quadrilateral:
            return "quadrilateral";
        case CellType::pyramid:
            return "pyramid";
        case CellType::prism:
            return "prism";
        case CellType::hexahedron:
            return "hexahedron";
        default:
            throw std::runtime_error("Unknown cell type.");
        }
    }

    CellType to_type(std::string_view cell)
    {
        if (cell == "point")
            return CellType::point;
        else if (cell == "interval" or cell == "interval2D"
            or cell == "interval3D")
            return CellType::interval;
        else if (cell == "triangle" or cell == "triangle3D")
            return CellType::triangle;
        else if (cell == "tetrahedron")
            return CellType::tetrahedron;
        else if (cell == "quadrilateral" or cell == "quadrilateral3D")
            return CellType::quadrilateral;
        else if (cell == "pyramid")
            return CellType::pyramid;
        else if (cell == "prism")
            return CellType::prism;
        else if (cell == "hexahedron")
            return CellType::hexahedron;
        else
            throw std::runtime_error(std::format("Unknown cell type ({})", cell));
    }

    graph::AdjacencyList<int> get_entity_vertices(CellType type, int dim)
    {
        std::vector<std::vector<int>> topology
            = basis::cell::topology(cell_type_to_basix_type(type))[dim];
        return graph::AdjacencyList<int>(topology);
    }

    graph::AdjacencyList<int> get_sub_entities(CellType type, int dim0, int dim1)
    {
        if (type == CellType::interval or type == CellType::point)
            return graph::AdjacencyList<int>(0);

        std::vector<std::vector<std::vector<int>>> connectivity
            = basis::cell::sub_entity_connectivity(
                cell_type_to_basix_type(type))[dim0];
        std::vector<std::vector<int>> subset;
        subset.reserve(connectivity.size());
        for (auto& row : connectivity)
            subset.emplace_back(row[dim1]);
        return graph::AdjacencyList<int>(subset);
    }

    int cell_num_entities(CellType type, int dim)
    {
        assert(dim <= 3);
        return basis::cell::num_sub_entities(cell_type_to_basix_type(type), dim);
    }

    bool is_simplex(CellType type) { return static_cast<int>(type) > 0; }

    int num_cell_vertices(CellType type)
    {
        return std::abs(static_cast<int>(type));
    }

    std::map<std::array<int, 2>, std::vector<std::set<int>>>
    cell_entity_closure(CellType cell_type)
    {
        const int tdim = cell_dim(cell_type);
        std::array<int, 4> num_entities;
        for (int i = 0; i <= tdim; ++i)
            num_entities[i] = cell_num_entities(cell_type, i);

        const graph::AdjacencyList<int> edge_v = get_entity_vertices(cell_type, 1);
        const graph::AdjacencyList<int> face_e = get_sub_entities(cell_type, 2, 1);

        std::map<std::array<int, 2>, std::vector<std::set<int>>> entity_closure;
        for (int dim = 0; dim <= tdim; ++dim) {
            for (int entity = 0; entity < num_entities[dim]; ++entity) {
                // Self
                entity_closure[{{dim, entity}}].resize(tdim + 1);
                entity_closure[{{dim, entity}}][dim].insert(entity);

                if (dim == 3) {
                    // All sub-entities of a 3D cell
                    for (int f = 0; f < num_entities[2]; ++f)
                        entity_closure[{{dim, entity}}][2].insert(f);
                    for (int e = 0; e < num_entities[1]; ++e)
                        entity_closure[{{dim, entity}}][1].insert(e);
                    for (int v = 0; v < num_entities[0]; ++v)
                        entity_closure[{{dim, entity}}][0].insert(v);
                }

                if (dim == 2) {
                    CellType face_type = cell_entity_type(cell_type, 2, entity);
                    const int num_edges = cell_num_entities(face_type, 1);
                    for (int e = 0; e < num_edges; ++e) {
                        const int edge_index = face_e.links(entity)[e];
                        entity_closure[{{dim, entity}}][1].insert(edge_index);
                        for (int v = 0; v < 2; ++v) {
                            entity_closure[{{dim, entity}}][0].insert(
                                edge_v.links(edge_index)[v]);
                        }
                    }
                }

                if (dim == 1) {
                    entity_closure[{{dim, entity}}][0].insert(
                        edge_v.links(entity)[0]);
                    entity_closure[{{dim, entity}}][0].insert(
                        edge_v.links(entity)[1]);
                }
            }
        }

        return entity_closure;
    }

    basis::cell::type cell_type_to_basix_type(CellType celltype)
    {
        switch (celltype) {
        case CellType::point:
            return basis::cell::type::point;
        case CellType::interval:
            return basis::cell::type::interval;
        case CellType::triangle:
            return basis::cell::type::triangle;
        case CellType::tetrahedron:
            return basis::cell::type::tetrahedron;
        case CellType::quadrilateral:
            return basis::cell::type::quadrilateral;
        case CellType::hexahedron:
            return basis::cell::type::hexahedron;
        case CellType::prism:
            return basis::cell::type::prism;
        case CellType::pyramid:
            return basis::cell::type::pyramid;
        default:
            throw std::runtime_error("Unrecognised cell type.");
        }
    }

    CellType cell_type_from_basix_type(basis::cell::type celltype)
    {
        switch (celltype) {
        case basis::cell::type::point:
            return CellType::point;
        case basis::cell::type::interval:
            return CellType::interval;
        case basis::cell::type::triangle:
            return CellType::triangle;
        case basis::cell::type::tetrahedron:
            return CellType::tetrahedron;
        case basis::cell::type::quadrilateral:
            return CellType::quadrilateral;
        case basis::cell::type::hexahedron:
            return CellType::hexahedron;
        case basis::cell::type::prism:
            return CellType::prism;
        case basis::cell::type::pyramid:
            return CellType::pyramid;
        default:
            throw std::runtime_error("Unrecognised cell type.");
        }
    }

} // namespace hellofem::mesh
