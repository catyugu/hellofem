// hellofem::mesh — cell type identifiers and reference-cell queries
// SPDX-License-Identifier: MIT

#pragma once

#include "basis/cell.h"
#include "graph/AdjacencyList.h"

#include <array>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace hellofem::mesh {

    /// Cell type identifier. Simplex cells have a positive index (see
    /// is_simplex); negative indices encode hypercube/pyramid/prism types.
    enum class CellType : std::int8_t {
        point = 1,
        interval = 2,
        triangle = 3,
        tetrahedron = 4,
        quadrilateral = -4,
        pyramid = -5,
        prism = -6,
        hexahedron = -8
    };

    /// String name for a cell type.
    std::string to_string(CellType type);

    /// Cell type from a string name.
    CellType to_type(std::string_view cell);

    /// Topological dimension of a cell type.
    inline int cell_dim(CellType type)
    {
        switch (type) {
        case CellType::point:
            return 0;
        case CellType::interval:
            return 1;
        case CellType::triangle:
        case CellType::quadrilateral:
            return 2;
        case CellType::tetrahedron:
        case CellType::hexahedron:
        case CellType::prism:
        case CellType::pyramid:
            return 3;
        default:
            throw std::runtime_error("Unsupported cell type");
        }
    }

    /// Type of facet `index` of a cell. Depends on the facet index only
    /// for prism and pyramid.
    inline CellType cell_facet_type(CellType type, int index)
    {
        switch (type) {
        case CellType::point:
            return CellType::point;
        case CellType::interval:
            return CellType::point;
        case CellType::triangle:
            return CellType::interval;
        case CellType::tetrahedron:
            return CellType::triangle;
        case CellType::quadrilateral:
            return CellType::interval;
        case CellType::pyramid:
            return index == 0 ? CellType::quadrilateral : CellType::triangle;
        case CellType::prism:
            return (index == 0 or index == 4) ? CellType::triangle
                                              : CellType::quadrilateral;
        case CellType::hexahedron:
            return CellType::quadrilateral;
        default:
            throw std::runtime_error("Unknown cell type.");
        }
    }

    /// Type of entity (dimension `d`, local `index`) of a cell.
    inline CellType cell_entity_type(CellType type, int d, int index)
    {
        if (int dim = cell_dim(type); d == dim)
            return type;
        else if (d == 1)
            return CellType::interval;
        else if (d == (dim - 1))
            return cell_facet_type(type, index);
        else
            return CellType::point;
    }

    /// Entities of dimension `dim`, each listed by its local vertex indices.
    graph::AdjacencyList<int> get_entity_vertices(CellType type, int dim);

    /// For each entity of dimension `dim0`, the entities of dimension
    /// `dim1` that make it up.
    graph::AdjacencyList<int> get_sub_entities(CellType type, int dim0, int dim1);

    /// Number of entities of a given dimension in a cell.
    int cell_num_entities(CellType type, int dim);

    /// True if the cell is a simplex.
    bool is_simplex(CellType type);

    /// Number of vertices of a cell type.
    int num_cell_vertices(CellType type);

    /// Closure of every cell entity: map from entity {dim_e, entity_e} to
    /// the lower-dimensional entities attached to it, keyed by their
    /// dimension.
    std::map<std::array<int, 2>, std::vector<std::set<int>>>
    cell_entity_closure(CellType cell_type);

    /// Convert a cell type to the basis cell type.
    basis::cell::type cell_type_to_basix_type(CellType celltype);

    /// Convert a basis cell type to a mesh cell type.
    CellType cell_type_from_basix_type(basis::cell::type celltype);

} // namespace hellofem::mesh
