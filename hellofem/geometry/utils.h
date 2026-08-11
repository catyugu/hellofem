// hellofem::geometry — bounding-box and GJK mesh queries
// SPDX-License-Identifier: MIT
//
// Collision, closest-entity and distance queries over a mesh, driven by
// a BoundingBoxTree. A box tree prunes the search; the GJK distance
// (gjk.h) gives the exact point-to-entity distance when a leaf is close
// enough to matter.

#pragma once

#include "BoundingBoxTree.h"
#include "gjk.h"
#include "graph/AdjacencyList.h"
#include "mesh/Mesh.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <numeric>
#include <optional>
#include <ranges>
#include <span>
#include <utility>
#include <vector>

namespace hellofem::geometry {

    /// A leaf node stores the entity index in both child slots.
    inline bool is_leaf(std::array<std::int32_t, 2> bbox)
    {
        return bbox[0] == bbox[1];
    }

    /// True if the point `x` lies in the box `b` (with a small relative
    /// tolerance on the box half-extent).
    template <std::floating_point T>
    bool point_in_bbox(std::array<T, 6> b, std::span<const T, 3> x)
    {
        constexpr T rtol = 1e-14;
        for (std::size_t i = 0; i < 3; ++i) {
            const T eps = rtol * (b[i + 3] - b[i]);
            if (x[i] < (b[i] - eps) or x[i] > (b[i + 3] + eps))
                return false;
        }
        return true;
    }

    /// True if boxes `a` and `b` overlap (relative tolerance).
    template <std::floating_point T>
    bool bbox_in_bbox(std::array<T, 6> a, std::array<T, 6> b)
    {
        constexpr T rtol = 1e-14;
        for (std::size_t i = 0; i < 3; ++i) {
            const T eps = rtol * (b[i + 3] - b[i]);
            if (a[i + 3] < (b[i] - eps) or a[i] > (b[i + 3] + eps))
                return false;
        }
        return true;
    }

    /// Squared distance from a point to a box (zero inside the box).
    template <std::floating_point T>
    T compute_squared_distance_bbox(std::array<T, 6> b, std::span<const T, 3> x)
    {
        T d = 0;
        for (std::size_t i = 0; i < 3; ++i) {
            if (x[i] < b[i]) {
                const T dx = x[i] - b[i];
                d += dx * dx;
            }
            else if (x[i] > b[i + 3]) {
                const T dx = x[i] - b[i + 3];
                d += dx * dx;
            }
        }
        return d;
    }

    /// Shortest vector from each of `entities` to the corresponding
    /// point in `points` (flat `(num_points, 3)`). The entity is
    /// represented by the convex hull of its vertices.
    template <std::floating_point T>
    std::vector<T> shortest_vector(const mesh::Mesh<T>& mesh, int dim,
        std::span<const std::int32_t> entities, std::span<const T> points)
    {
        std::span<const T> geom_dofs = mesh.geometry().x();
        const int tdim = mesh.topology()->dim();

        std::vector<T> shortest;
        shortest.reserve(3 * entities.size());

        // For cells (dim == tdim) the vertices come straight from the
        // geometry dofmap row; for sub-entities, go through the attached
        // cell and the element dof layout of the entity.
        auto x_dofmap = mesh.geometry().dofmaps().front();
        const fem::CoordinateElement<T>& cmap = mesh.geometry().cmaps().front();
        if (dim == tdim) {
            for (std::size_t e = 0; e < entities.size(); ++e) {
                assert(entities[e] >= 0);
                const std::int32_t cell = entities[e];
                std::vector<T> nodes(3 * x_dofmap.extent(1));
                for (std::size_t i = 0; i < x_dofmap.extent(1); ++i)
                    for (int k = 0; k < 3; ++k)
                        nodes[3 * i + k]
                            = geom_dofs[3 * x_dofmap(cell, i) + k];
                auto d = compute_distance_gjk<T>(
                    points.subspan(3 * e, 3), nodes);
                shortest.insert(shortest.end(), d.begin(), d.end());
            }
        }
        else {
            auto* topology = mesh.topology_mutable().get();
            topology->create_connectivity(dim, tdim);
            topology->create_connectivity(tdim, dim);
            auto e_to_c = topology->connectivity(dim, tdim);
            auto c_to_e = topology->connectivity(tdim, dim);
            auto layout = cmap.create_dof_layout();

            for (std::size_t e = 0; e < entities.size(); ++e) {
                const std::int32_t index = entities[e];
                assert(e_to_c->num_links(index) > 0);
                const std::int32_t c = e_to_c->links(index)[0];

                auto cell_entities = c_to_e->links(c);
                auto it = std::find(cell_entities.begin(),
                    cell_entities.end(), index);
                assert(it != cell_entities.end());
                const std::int32_t local
                    = static_cast<std::int32_t>(it - cell_entities.begin());

                const std::vector<int> entity_dofs
                    = layout.entity_closure_dofs(dim, local);
                std::vector<T> nodes(3 * entity_dofs.size());
                for (std::size_t i = 0; i < entity_dofs.size(); ++i)
                    for (int k = 0; k < 3; ++k)
                        nodes[3 * i + k]
                            = geom_dofs[3 * x_dofmap(c, entity_dofs[i]) + k];

                auto d = compute_distance_gjk<T>(
                    points.subspan(3 * e, 3), nodes);
                shortest.insert(shortest.end(), d.begin(), d.end());
            }
        }

        return shortest;
    }

    /// Squared shortest distance from each of `entities` to the
    /// corresponding point.
    template <std::floating_point T, typename R>
        requires std::ranges::contiguous_range<R>
        and std::same_as<std::ranges::range_value_t<R>, T>
    std::vector<T> squared_distance(const mesh::Mesh<T>& mesh, int dim,
        std::span<const std::int32_t> entities, const R& points)
    {
        std::span<const T> pts(points);
        std::vector<T> v = shortest_vector(mesh, dim, entities, pts);
        std::vector<T> d(v.size() / 3, 0);
        for (std::size_t i = 0; i < d.size(); ++i)
            for (int k = 0; k < 3; ++k)
                d[i] += v[3 * i + k] * v[3 * i + k];
        return d;
    }

    namespace impl {

        /// Recursively find the entity whose box (or, for point clouds,
        /// whose point) is closest to `point`, pruning subtrees whose box
        /// distance exceeds the running best `R2`. Returns
        /// `{closest_entity, best_R2}`.
        template <std::floating_point T>
        std::pair<std::int32_t, T> compute_closest_entity(
            const BoundingBoxTree<T>& tree, std::span<const T, 3> point,
            std::int32_t node, const mesh::Mesh<T>& mesh,
            std::int32_t closest_entity, T R2)
        {
            const std::array<std::int32_t, 2> bbox = tree.bbox(node);
            T r2;
            if (is_leaf(bbox)) {
                if (tree.tdim() == 0) {
                    // Point cloud: the leaf box is the point itself.
                    std::array<T, 6> diff = tree.get_bbox(node);
                    r2 = 0;
                    for (int k = 0; k < 3; ++k) {
                        const T d = diff[k] - point[k];
                        r2 += d * d;
                    }
                }
                else {
                    r2 = compute_squared_distance_bbox<T>(
                        tree.get_bbox(node), point);
                    // Box closer than best -> exact GJK distance.
                    if (r2 <= R2) {
                        const std::array<T, 3> p {point[0], point[1], point[2]};
                        r2 = squared_distance<T>(mesh, tree.tdim(),
                            std::span<const std::int32_t>(&bbox[1], 1), p)
                                 .front();
                    }
                }

                if (r2 <= R2) {
                    closest_entity = bbox[1];
                    R2 = r2;
                }
                return {closest_entity, R2};
            }

            r2 = compute_squared_distance_bbox<T>(tree.get_bbox(node), point);
            if (r2 > R2)
                return {closest_entity, R2};

            // Use R2 (not r2) as the radius for the children: a box can
            // be closer than the entity it contains.
            auto p0 = compute_closest_entity<T>(
                tree, point, bbox[0], mesh, closest_entity, R2);
            return compute_closest_entity<T>(
                tree, point, bbox[1], mesh, p0.first, p0.second);
        }

        /// Collect the leaves whose box contains `p` (iterative stack
        /// traversal).
        template <std::floating_point T>
        void compute_collisions_point(const BoundingBoxTree<T>& tree,
            std::span<const T, 3> p, std::vector<std::int32_t>& entities)
        {
            std::deque<std::int32_t> stack;
            std::int32_t next = tree.num_bboxes() - 1;
            std::span<const T> coords = tree.bbox_coordinates();
            auto view = [&coords](std::size_t node) {
                std::array<T, 6> b;
                std::copy_n(coords.data() + 6 * node, 6, b.begin());
                return b;
            };

            while (next != -1) {
                const std::array bbox = tree.bbox(next);
                if (is_leaf(bbox) and point_in_bbox<T>(view(next), p)) {
                    entities.push_back(bbox[1]);
                    next = -1;
                }
                else {
                    const bool left = point_in_bbox<T>(view(bbox[0]), p);
                    const bool right = point_in_bbox<T>(view(bbox[1]), p);
                    if (left and right) {
                        stack.push_back(bbox[1]);
                        next = bbox[0];
                    }
                    else if (left)
                        next = bbox[0];
                    else if (right)
                        next = bbox[1];
                    else
                        next = -1;
                }

                if (next == -1 and not stack.empty()) {
                    next = stack.back();
                    stack.pop_back();
                }
            }
        }

        /// Recursively collect pairs of colliding leaves of two trees.
        template <std::floating_point T>
        void compute_collisions_tree(const BoundingBoxTree<T>& A,
            const BoundingBoxTree<T>& B, std::int32_t node_A,
            std::int32_t node_B, std::vector<std::int32_t>& entities)
        {
            if (!bbox_in_bbox<T>(A.get_bbox(node_A), B.get_bbox(node_B)))
                return;

            const auto bbox_A = A.bbox(node_A);
            const auto bbox_B = B.bbox(node_B);
            const bool leaf_A = is_leaf(bbox_A);
            const bool leaf_B = is_leaf(bbox_B);

            if (leaf_A and leaf_B) {
                entities.push_back(bbox_A[1]);
                entities.push_back(bbox_B[1]);
            }
            else if (leaf_A) {
                compute_collisions_tree<T>(
                    A, B, node_A, bbox_B[0], entities);
                compute_collisions_tree<T>(
                    A, B, node_A, bbox_B[1], entities);
            }
            else if (leaf_B) {
                compute_collisions_tree<T>(
                    A, B, bbox_A[0], node_B, entities);
                compute_collisions_tree<T>(
                    A, B, bbox_A[1], node_B, entities);
            }
            else if (node_A > node_B) {
                // Descend the larger tree first (nodes are appended
                // post-order, so larger indices own larger subtrees).
                compute_collisions_tree<T>(
                    A, B, bbox_A[0], node_B, entities);
                compute_collisions_tree<T>(
                    A, B, bbox_A[1], node_B, entities);
            }
            else {
                compute_collisions_tree<T>(
                    A, B, node_A, bbox_B[0], entities);
                compute_collisions_tree<T>(
                    A, B, node_A, bbox_B[1], entities);
            }
        }

    } // namespace impl

    /// Midpoints of mesh entities, flat `(num_entities, 3)`, computed as
    /// the average of each entity's vertex coordinates.
    template <std::floating_point T>
    std::vector<T> compute_midpoints(const mesh::Mesh<T>& mesh, int dim,
        std::span<const std::int32_t> entities)
    {
        std::span<const T> xg = mesh.geometry().x();
        std::vector<T> midpoints(3 * entities.size(), 0);
        for (std::size_t e = 0; e < entities.size(); ++e) {
            auto verts = mesh.topology()->connectivity(dim, 0)->links(entities[e]);
            for (auto v : verts)
                for (int k = 0; k < 3; ++k)
                    midpoints[3 * e + k] += xg[3 * v + k];
            for (int k = 0; k < 3; ++k)
                midpoints[3 * e + k] /= static_cast<T>(verts.size());
        }
        return midpoints;
    }

    /// A point-cloud tree over the midpoints of a subset of entities.
    template <std::floating_point T>
    BoundingBoxTree<T> create_midpoint_tree(const mesh::Mesh<T>& mesh,
        int tdim, std::span<const std::int32_t> entities)
    {
        const std::vector<T> midpoints = compute_midpoints<T>(mesh, tdim, entities);
        std::vector<std::pair<std::array<T, 3>, std::int32_t>> points(
            entities.size());
        for (std::size_t i = 0; i < points.size(); ++i) {
            for (int k = 0; k < 3; ++k)
                points[i].first[k] = midpoints[3 * i + k];
            points[i].second = entities[i];
        }
        return BoundingBoxTree<T>(std::move(points));
    }

    /// Pairs of colliding leaves of two trees, flattened `(num_hits, 2)`.
    template <std::floating_point T>
    std::vector<std::int32_t> compute_collisions(
        const BoundingBoxTree<T>& tree0, const BoundingBoxTree<T>& tree1)
    {
        std::vector<std::int32_t> entities;
        if (tree0.num_bboxes() > 0 and tree1.num_bboxes() > 0)
            impl::compute_collisions_tree<T>(tree0, tree1,
                tree0.num_bboxes() - 1, tree1.num_bboxes() - 1, entities);
        return entities;
    }

    /// For each point, the leaves whose box contains it.
    template <std::floating_point T, typename R>
        requires std::ranges::contiguous_range<R>
        and std::same_as<std::ranges::range_value_t<R>, T>
    graph::AdjacencyList<std::int32_t> compute_collisions(
        const BoundingBoxTree<T>& tree, const R& points)
    {
        std::span<const T> pts(points);
        const std::size_t num_points = pts.size() / 3;
        std::vector<std::int32_t> entities, offsets(num_points + 1, 0);
        if (tree.num_bboxes() > 0) {
            entities.reserve(num_points);
            for (std::size_t p = 0; p < num_points; ++p) {
                impl::compute_collisions_point<T>(tree,
                    std::span<const T, 3>(pts.data() + 3 * p, 3), entities);
                offsets[p + 1] = static_cast<std::int32_t>(entities.size());
            }
        }
        return graph::AdjacencyList(std::move(entities), std::move(offsets));
    }

    /// For each point, the mesh entity closest to it. A midpoint tree
    /// supplies the initial guess; the entity tree prunes by box distance
    /// and refines with GJK.
    template <std::floating_point T, typename R>
        requires std::ranges::contiguous_range<R>
        and std::same_as<std::ranges::range_value_t<R>, T>
    std::vector<std::int32_t> compute_closest_entity(
        const BoundingBoxTree<T>& tree, const BoundingBoxTree<T>& midpoint_tree,
        const mesh::Mesh<T>& mesh, const R& points)
    {
        std::span<const T> pts(points);
        if (tree.num_bboxes() == 0)
            return std::vector<std::int32_t>(pts.size() / 3, -1);

        std::vector<std::int32_t> entities;
        entities.reserve(pts.size() / 3);
        for (std::size_t i = 0; i < pts.size() / 3; ++i) {
            // Initial guess: a leaf of the midpoint tree and its
            // distance to the point.
            const std::array leaf0 = midpoint_tree.bbox(0);
            assert(is_leaf(leaf0));
            std::array<T, 6> diff = midpoint_tree.get_bbox(0);
            T R2 = 0;
            for (int k = 0; k < 3; ++k) {
                const T d = diff[k] - pts[3 * i + k];
                R2 += d * d;
            }

            auto [m_index, m_distance2] = impl::compute_closest_entity<T>(
                midpoint_tree, std::span<const T, 3>(pts.data() + 3 * i, 3),
                midpoint_tree.num_bboxes() - 1, mesh, leaf0[0], R2);

            const auto [index, distance2] = impl::compute_closest_entity<T>(
                tree, std::span<const T, 3>(pts.data() + 3 * i, 3),
                tree.num_bboxes() - 1, mesh, m_index, m_distance2);
            (void)distance2;

            entities.push_back(index);
        }
        return entities;
    }

    /// For each point, the candidate cells that truly collide with it
    /// (GJK distance below `1e-12`), refining a coarse candidate list.
    template <std::floating_point T, typename R>
        requires std::ranges::contiguous_range<R>
        and std::same_as<std::ranges::range_value_t<R>, T>
    graph::AdjacencyList<std::int32_t> compute_colliding_cells(
        const mesh::Mesh<T>& mesh,
        const graph::AdjacencyList<std::int32_t>& candidate_cells,
        const R& points)
    {
        std::span<const T> pts(points);
        std::vector<std::int32_t> colliding, offsets = {0};
        offsets.reserve(candidate_cells.num_nodes() + 1);
        constexpr T eps2 = 1e-12;
        const int tdim = mesh.topology()->dim();

        for (std::int32_t i = 0; i < candidate_cells.num_nodes(); ++i) {
            auto cells = candidate_cells.links(i);
            std::vector<T> _point(3 * cells.size());
            for (std::size_t j = 0; j < cells.size(); ++j)
                for (int k = 0; k < 3; ++k)
                    _point[3 * j + k] = pts[3 * i + k];

            std::vector<T> distances_sq
                = squared_distance<T>(mesh, tdim, cells, _point);
            for (std::size_t j = 0; j < cells.size(); ++j)
                if (distances_sq[j] < eps2)
                    colliding.push_back(cells[j]);

            offsets.push_back(static_cast<std::int32_t>(colliding.size()));
        }
        return graph::AdjacencyList(std::move(colliding), std::move(offsets));
    }

} // namespace hellofem::geometry
