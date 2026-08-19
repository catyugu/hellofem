// hellofem::mesh — mesh utilities
// SPDX-License-Identifier: MIT

#pragma once

#include "Mesh.h"
#include "Topology.h"
#include "mesh/cell_types.h"

#include <algorithm>
#include <array>
#include <concepts>
#include <cstdint>
#include <numeric>
#include <span>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace hellofem::mesh {

    /// For each entity in `entities` (of dimension `d0`), collect the
    /// distinct incident entities of dimension `d1` via the `d0 -> d1`
    /// connectivity. Requires the entities of both dimensions and the
    /// connectivity to exist.
    std::vector<std::int32_t>
    compute_incident_entities(const Topology& topology,
        std::span<const std::int32_t> entities, int d0, int d1);

    /// Requirement on geometry-marking functions: a marker takes the
    /// `(3, num_points)` coordinate array of entity vertices and returns
    /// one flag per point.
    template <typename Fn, typename T>
    concept MarkerFn = std::is_invocable_r<std::vector<std::int8_t>, Fn,
        md::mdspan<const T, md::extents<std::size_t, 3, md::dynamic_extent>>>::value;

    /// Indices of the facets of facet type `facet_type_idx` that are
    /// attached to exactly one cell (the mesh boundary).
    ///
    /// @param[in] topology Mesh topology.
    /// @param[in] facet_type_idx Index of the facet type in
    /// `topology.entity_types(tdim - 1)`.
    std::vector<std::int32_t>
    exterior_facet_indices(const Topology& topology, int facet_type_idx);

    /// Indices of the exterior facets of the (sole) facet type.
    std::vector<std::int32_t> exterior_facet_indices(const Topology& topology);

    /// Coordinates of every vertex of the mesh, stored column-major with
    /// shape `(3, num_vertices)` (the `j`-th column is vertex `j`).
    template <std::floating_point T>
    std::pair<std::vector<T>, std::array<std::size_t, 2>>
    compute_vertex_coords(const mesh::Mesh<T>& mesh);

    /// Coordinates of the midpoint of each entity in `entities`, stored
    /// row-major with 3 entries per entity.
    template <std::floating_point T>
    std::vector<T> compute_midpoints(const mesh::Mesh<T>& mesh, int dim,
        std::span<const std::int32_t> entities);

    /// Indices of all entities of dimension `dim` whose every vertex is
    /// marked by `marker` (indices local to the process).
    ///
    /// @param[in] entity_type_idx Index of the entity type in
    /// `topology.entity_types(dim)`. The single-type overload uses 0.
    template <std::floating_point T, MarkerFn<T> U>
    std::vector<std::int32_t>
    locate_entities(const mesh::Mesh<T>& mesh, int dim, U marker,
        int entity_type_idx);

    /// Single-type overload of @ref locate_entities (entity type index 0).
    template <std::floating_point T, MarkerFn<T> U>
    std::vector<std::int32_t>
    locate_entities(const mesh::Mesh<T>& mesh, int dim, U marker);

    /// Indices of all boundary entities of dimension `dim` (must be less
    /// than the topological dimension) whose every vertex is marked by
    /// `marker`.
    template <std::floating_point T, MarkerFn<T> U>
    std::vector<std::int32_t>
    locate_entities_boundary(const mesh::Mesh<T>& mesh, int dim, U marker);

    /// Geometry degrees-of-freedom of the closure of each entity, stored
    /// row-major with shape `(num_entities, num_dofs_per_entity)`.
    ///
    /// @param[in] permute If true, permute the dofs to be consistent
    /// with the mesh orientation of the entities (requires
    /// `create_entity_permutations` to have been called).
    /// @pre Connectivities `dim -> tdim` and `tdim -> dim` exist.
    template <std::floating_point T>
    std::pair<std::vector<std::int32_t>, std::array<std::size_t, 2>>
    entities_to_geometry(const mesh::Mesh<T>& mesh, int dim,
        std::span<const std::int32_t> entities, bool permute = false);

    /// Create a mesh with affine P1 geometry from cell-to-vertex
    /// connectivity and vertex coordinates.
    ///
    /// @param[in] cells Cell-to-vertex connectivity using *global* vertex
    /// indices (row-major, `num_cells * num_vertices` entries), ordered
    /// according to the `cell_type` reference ordering.
    /// @param[in] cell_type Cell shape.
    /// @param[in] x Vertex coordinates (row-major, `(num_vertices, gdim)`).
    /// @param[in] gdim Geometric dimension.
    template <typename U>
    Mesh<typename std::remove_cvref_t<typename U::value_type>> create_mesh(
        std::span<const std::int64_t> cells, CellType cell_type, const U& x,
        int gdim);

    // ------------------------------------------------------------------ //
    //                         Inline implementations                      //
    // ------------------------------------------------------------------ //

    namespace impl {

        /// Map each mesh vertex to its geometry 'node' index. The
        /// geometry dofmap lists the cell geometry dofs in basix order
        /// (vertices first), so the `i`-th vertex of a cell maps to its
        /// `i`-th geometry dof.
        template <std::floating_point T>
        std::vector<std::int32_t>
        vertex_to_geometry_node(const mesh::Mesh<T>& mesh)
        {
            auto topology = mesh.topology();
            const int tdim = topology->dim();
            std::vector<std::int32_t> vertex_to_node(
                topology->index_map(0)->size_local(), -1);
            for (std::size_t cell_type_idx = 0;
                 cell_type_idx < topology->entity_types(tdim).size();
                 ++cell_type_idx) {
                auto x_dofmap
                    = mesh.geometry().dofmaps().at(cell_type_idx);
                auto c_to_v = topology->connectivity(
                    {tdim, static_cast<int>(cell_type_idx)}, {0, 0});
                for (std::int32_t c = 0; c < c_to_v->num_nodes(); ++c) {
                    auto vertices = c_to_v->links(c);
                    for (std::size_t i = 0; i < vertices.size(); ++i)
                        vertex_to_node[vertices[i]] = x_dofmap(c, i);
                }
            }
            return vertex_to_node;
        }

        /// Entities (of dimension `dim`) attached to the given boundary
        /// facets, plus the coordinates of the boundary vertices and a
        /// map from mesh vertex index to position in the coordinate array.
        template <std::floating_point T>
        std::tuple<std::vector<std::int32_t>, std::vector<T>,
            std::vector<std::int32_t>>
        compute_vertex_coords_boundary(const mesh::Mesh<T>& mesh, int dim,
            std::span<const std::int32_t> facets)
        {
            auto topology = mesh.topology_mutable();
            const int tdim = topology->dim();
            if (dim == tdim)
                throw std::runtime_error(
                    "Cannot locate boundary entities for cells.");

            topology->create_connectivity(tdim - 1, 0);
            topology->create_connectivity(tdim - 1, dim);
            auto f_to_v = topology->connectivity(tdim - 1, 0);
            auto f_to_e = topology->connectivity(tdim - 1, dim);

            std::vector<std::int32_t> vertices, entities;
            for (auto f : facets) {
                for (auto v : f_to_v->links(f))
                    vertices.push_back(v);
                for (auto e : f_to_e->links(f))
                    entities.push_back(e);
            }
            std::ranges::sort(vertices);
            vertices.erase(std::ranges::unique(vertices).begin(),
                vertices.end());
            std::ranges::sort(entities);
            entities.erase(std::ranges::unique(entities).begin(),
                entities.end());

            // Map mesh vertices to positions in the boundary coordinate array.
            const auto vertex_to_node = impl::vertex_to_geometry_node(mesh);
            std::span<const T> x_nodes = mesh.geometry().x();
            std::vector<std::int32_t> vertex_to_pos(
                topology->index_map(0)->size_local(), -1);
            std::vector<T> x_vertices(3 * vertices.size(), 0);
            for (std::size_t i = 0; i < vertices.size(); ++i) {
                const std::int32_t v = vertices[i];
                vertex_to_pos[v] = static_cast<std::int32_t>(i);
                const std::int32_t pos = 3 * vertex_to_node[v];
                for (std::size_t j = 0; j < 3; ++j)
                    x_vertices[j * vertices.size() + i] = x_nodes[pos + j];
            }

            return {std::move(entities), std::move(x_vertices),
                std::move(vertex_to_pos)};
        }

    } // namespace impl

    /// Coordinates of every vertex of the mesh, stored column-major with
    /// shape `(3, num_vertices)`.
    ///
    /// For P1 meshes, returns only the coordinate of each geometric vertex.
    /// For P2+ meshes (with mid-edge nodes), returns coordinates of all
    /// geometry nodes in order (vertex + mid-edge nodes), so that callers
    /// can evaluate functions at any geometry node.
    template <std::floating_point T>
    std::pair<std::vector<T>, std::array<std::size_t, 2>>
    compute_vertex_coords(const mesh::Mesh<T>& mesh)
    {
        auto topology = mesh.topology();
        const std::int32_t num_vertices
            = topology->index_map(0)->size_local();

        const auto vertex_to_node = impl::vertex_to_geometry_node(mesh);
        std::span<const T> x_nodes = mesh.geometry().x();
        const std::size_t num_geom_nodes = mesh.geometry().index_map()->size_local();

        // P1 case: every topology vertex maps to a unique geometry node.
        if (num_vertices == static_cast<std::int32_t>(num_geom_nodes) &&
            std::all_of(vertex_to_node.begin(), vertex_to_node.end(),
                [](std::int32_t v) { return v >= 0; })) {
            std::vector<T> x_vertices(3 * num_vertices, 0);
            for (std::int32_t i = 0; i < num_vertices; ++i) {
                const std::int32_t pos = 3 * vertex_to_node[i];
                for (std::size_t j = 0; j < 3; ++j)
                    x_vertices[j * num_vertices + i] = x_nodes[pos + j];
            }
            return {std::move(x_vertices), {3, static_cast<std::size_t>(num_vertices)}};
        }

        // P2+ fallback: only return coordinates of topology vertices that
        // map to geometry nodes (skip interior/mid-edge nodes that have no
        // corresponding topology vertex).
        std::vector<T> x_vertices(3 * num_vertices, 0);
        std::int32_t nv_written = 0;
        for (std::int32_t i = 0; i < num_vertices; ++i) {
            const std::int32_t pos = vertex_to_node[i];
            if (pos >= 0) {
                for (std::size_t j = 0; j < 3; ++j)
                    x_vertices[j * num_vertices + nv_written] = x_nodes[3 * pos + j];
                ++nv_written;
            }
        }
        return {std::move(x_vertices), {3, static_cast<std::size_t>(nv_written)}};
    }

    template <std::floating_point T>
    std::vector<T> compute_midpoints(const mesh::Mesh<T>& mesh, int dim,
        std::span<const std::int32_t> entities)
    {
        if (entities.empty())
            return {};

        std::span<const T> x = mesh.geometry().x();
        const auto [e_to_g, eshape]
            = entities_to_geometry(mesh, dim, entities, false);

        std::vector<T> x_mid(entities.size() * 3, 0);
        for (std::size_t e = 0; e < entities.size(); ++e) {
            std::span<T, 3> p(x_mid.data() + 3 * e, 3);
            std::span<const std::int32_t> rows(
                e_to_g.data() + e * eshape[1], eshape[1]);
            for (std::int32_t row : rows) {
                std::span<const T, 3> xg(x.data() + 3 * row, 3);
                for (std::size_t j = 0; j < 3; ++j)
                    p[j] += xg[j] / rows.size();
            }
        }

        return x_mid;
    }

    template <std::floating_point T>
    std::pair<std::vector<std::int32_t>, std::array<std::size_t, 2>>
    entities_to_geometry(const mesh::Mesh<T>& mesh, int dim,
        std::span<const std::int32_t> entities, bool permute)
    {
        auto topology = mesh.topology();
        const int tdim = topology->dim();
        const Geometry<T>& geometry = mesh.geometry();
        auto xdofs = geometry.dofmaps().front();
        const fem::CoordinateElement<T>& coord_ele = geometry.cmaps().front();
        const fem::ElementDofLayout layout = coord_ele.create_dof_layout();
        const std::vector<std::vector<std::vector<int>>>& closure_dofs_all
            = layout.entity_closure_dofs_all();

        std::vector<std::int32_t> entity_xdofs;
        const auto eshape = std::array<std::size_t, 2> {
            entities.size(),
            layout.entity_closure_dofs(dim, 0).size()};
        entity_xdofs.reserve(eshape[0] * eshape[1]);

        if (dim == tdim) {
            for (std::int32_t c : entities) {
                for (std::int32_t entity_dof : closure_dofs_all[tdim][0])
                    entity_xdofs.push_back(xdofs(c, entity_dof));
            }
            return {std::move(entity_xdofs), eshape};
        }

        auto e_to_c = topology->connectivity(dim, tdim);
        auto c_to_e = topology->connectivity(tdim, dim);
        if (!e_to_c or !c_to_e)
            throw std::runtime_error(
                "Missing connectivity for entities_to_geometry.");

        std::span<const std::uint32_t> cell_info;
        if (permute)
            cell_info = std::span(topology->get_cell_permutation_info());

        const CellType cell_type = topology->cell_type();
        for (std::int32_t e : entities) {
            assert(!e_to_c->links(e).empty());
            const std::int32_t c = e_to_c->links(e).front();
            auto cell_entities = c_to_e->links(c);
            auto it = std::find(cell_entities.begin(), cell_entities.end(), e);
            assert(it != cell_entities.end());
            const std::size_t local_entity
                = std::distance(cell_entities.begin(), it);

            std::vector<std::int32_t> closure_dofs(
                closure_dofs_all[dim][local_entity]);
            if (permute) {
                const CellType entity_type
                    = cell_entity_type(cell_type, dim, local_entity);
                coord_ele.permute_subentity_closure(closure_dofs, cell_info[c],
                    entity_type, local_entity);
            }

            for (std::int32_t entity_dof : closure_dofs)
                entity_xdofs.push_back(xdofs(c, entity_dof));
        }

        return {std::move(entity_xdofs), eshape};
    }

    template <std::floating_point T, MarkerFn<T> U>
    std::vector<std::int32_t>
    locate_entities(const mesh::Mesh<T>& mesh, int dim, U marker,
        int entity_type_idx)
    {
        auto topology = mesh.topology_mutable();

        using cmdspan3x_t
            = md::mdspan<const T, md::extents<std::size_t, 3, md::dynamic_extent>>;

        // Coordinates of the vertices of all entities of dimension `dim`.
        topology->create_entities(dim);
        auto e_to_v = topology->connectivity({dim, entity_type_idx}, {0, 0});
        if (!e_to_v)
            throw std::runtime_error("Missing entity-vertex connectivity.");

        std::vector<std::int32_t> vertices;
        for (std::int32_t e = 0; e < e_to_v->num_nodes(); ++e)
            for (auto v : e_to_v->links(e))
                vertices.push_back(v);
        std::ranges::sort(vertices);
        vertices.erase(std::ranges::unique(vertices).begin(), vertices.end());

        const auto vertex_to_node = impl::vertex_to_geometry_node(mesh);
        std::span<const T> x_nodes = mesh.geometry().x();
        std::vector<T> x_vertices(3 * vertices.size(), 0);
        std::vector<std::int32_t> vertex_to_pos(
            topology->index_map(0)->size_local(), -1);
        for (std::size_t i = 0; i < vertices.size(); ++i) {
            const std::int32_t v = vertices[i];
            vertex_to_pos[v] = static_cast<std::int32_t>(i);
            const std::int32_t pos = 3 * vertex_to_node[v];
            for (std::size_t j = 0; j < 3; ++j)
                x_vertices[j * vertices.size() + i] = x_nodes[pos + j];
        }

        std::vector<std::int8_t> marked
            = marker(cmdspan3x_t(x_vertices.data(), 3, vertices.size()));
        if (marked.size() != vertices.size())
            throw std::runtime_error(
                "Marker function returned the wrong number of flags.");

        std::vector<std::int32_t> entities;
        for (std::int32_t e = 0; e < e_to_v->num_nodes(); ++e) {
            bool all_marked = true;
            for (auto v : e_to_v->links(e)) {
                if (!marked[vertex_to_pos[v]]) {
                    all_marked = false;
                    break;
                }
            }
            if (all_marked)
                entities.push_back(e);
        }

        return entities;
    }

    template <std::floating_point T, MarkerFn<T> U>
    std::vector<std::int32_t>
    locate_entities(const mesh::Mesh<T>& mesh, int dim, U marker)
    {
        if (mesh.topology()->entity_types(dim).size() > 1)
            throw std::runtime_error(
                "Multiple entity types of this dimension. Specify entity "
                "type index");
        return locate_entities(mesh, dim, marker, 0);
    }

    template <std::floating_point T, MarkerFn<T> U>
    std::vector<std::int32_t>
    locate_entities_boundary(const mesh::Mesh<T>& mesh, int dim, U marker)
    {
        auto topology = mesh.topology_mutable();
        const int tdim = topology->dim();
        if (dim == tdim)
            throw std::runtime_error(
                "Cannot use locate_entities_boundary for cells.");

        topology->create_entities(tdim - 1);
        topology->create_connectivity(tdim - 1, tdim);
        const std::vector<std::int32_t> boundary_facets
            = exterior_facet_indices(*topology);

        using cmdspan3x_t
            = md::mdspan<const T, md::extents<std::size_t, 3, md::dynamic_extent>>;

        auto [facet_entities, xdata, vertex_to_pos]
            = impl::compute_vertex_coords_boundary(mesh, dim, boundary_facets);
        std::vector<std::int8_t> marked
            = marker(cmdspan3x_t(xdata.data(), 3, xdata.size() / 3));
        if (marked.size() != xdata.size() / 3)
            throw std::runtime_error(
                "Marker function returned the wrong number of flags.");

        topology->create_entities(dim);
        auto e_to_v = topology->connectivity(dim, 0);
        std::vector<std::int32_t> entities;
        for (std::int32_t e : facet_entities) {
            bool all_marked = true;
            for (auto v : e_to_v->links(e)) {
                const std::int32_t pos = vertex_to_pos[v];
                if (!marked[pos]) {
                    all_marked = false;
                    break;
                }
            }
            if (all_marked)
                entities.push_back(e);
        }

        return entities;
    }

    template <typename U>
    Mesh<typename std::remove_cvref_t<typename U::value_type>> create_mesh(
        std::span<const std::int64_t> cells, CellType cell_type, const U& x,
        int gdim)
    {
        using T = typename std::remove_cvref_t<typename U::value_type>;
        const std::size_t num_vertices_per_cell = mesh::cell_num_entities(cell_type, 0);
        if (cells.size() % num_vertices_per_cell != 0)
            throw std::invalid_argument(
                "create_mesh: cell data is not a multiple of the cell size.");
        const std::int64_t num_cells
            = cells.size() / num_vertices_per_cell;

        // Original cell index (identity: input order is preserved).
        std::vector<std::int64_t> orig(num_cells);
        std::iota(orig.begin(), orig.end(), 0);

        auto topology = std::make_shared<Topology>(create_topology(cells,
            std::span<const std::int64_t>(orig), cell_type, 1));

        // Affine geometry: one geometry node per vertex, coordinates from
        // `x`; node global indices are the (contiguous) vertex indices.
        const std::size_t num_nodes = x.size() / gdim;
        std::vector<std::int64_t> nodes(num_nodes);
        std::iota(nodes.begin(), nodes.end(), 0);

        fem::CoordinateElement<T> coord_el(cell_type, 1);
        auto geometry = create_geometry(*topology,
            std::vector<fem::CoordinateElement<T>> {coord_el},
            std::span<const std::int64_t>(nodes), cells, x, gdim);

        return Mesh<T>(topology, std::move(geometry));
    }

} // namespace hellofem::mesh
