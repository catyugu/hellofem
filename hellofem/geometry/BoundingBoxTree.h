// hellofem::geometry — axis-aligned bounding box tree
// SPDX-License-Identifier: MIT
//
// A binary tree over the bounding boxes of a set of mesh entities (or a
// point cloud). Node `i` occupies slots [2i, 2i+1] of the flat `_bboxes`
// array: an internal node stores its two child node indices, a leaf
// stores the entity index in both slots (equal -> leaf). Each node's box
// is 6 coordinates {lower[3], upper[3]} stored at 6*i. Leaves are
// appended first, the root last, so `num_bboxes() - 1` is the root and a
// full tree has 2*N - 1 nodes.

#pragma once

#include "mesh/Geometry.h"
#include "mesh/Mesh.h"
#include "mesh/Topology.h"
#include "mesh/cell_types.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace hellofem::geometry {

    namespace impl_bb {

        /// Bounding box of a mesh entity: the min/max of its vertices'
        /// coordinates, taken from the (flat, 3-per-point) geometry array.
        /// For a cell the vertices are its geometry-dofmap points; for a
        /// lower-dimensional entity, the vertices of an attached cell that
        /// lie on the entity. A linear geometry has one geometry point per
        /// vertex, so cell-local geometry dof `i` is cell-local vertex `i`
        /// and `x_dofmap(c, lv)` maps a local vertex to its geometry point.
        template <std::floating_point T>
        std::array<T, 6> compute_bbox_of_entity(const mesh::Mesh<T>& mesh,
            int dim, std::int32_t index)
        {
            std::span<const T> xg = mesh.geometry().x();
            auto x_dofmap = mesh.geometry().dofmaps().front();
            const int tdim = mesh.topology()->dim();

            std::vector<std::int32_t> points;
            if (dim == tdim) {
                const std::size_t ndofs = x_dofmap.extent(1);
                for (std::size_t i = 0; i < ndofs; ++i)
                    points.push_back(x_dofmap(index, i));
            }
            else {
                // Attached cell and the entity's local index in it.
                auto* topology = mesh.topology_mutable().get();
                topology->create_connectivity(dim, tdim);
                topology->create_connectivity(tdim, dim);
                auto e_to_c = topology->connectivity(dim, tdim);
                auto c_to_e = topology->connectivity(tdim, dim);
                const std::int32_t c = e_to_c->links(index)[0];
                auto cell_entities = c_to_e->links(c);
                auto it = std::find(cell_entities.begin(), cell_entities.end(),
                    index);
                const std::int32_t local
                    = static_cast<std::int32_t>(it - cell_entities.begin());

                // Local vertices of the entity in the cell, mapped to
                // geometry points through the cell's geometry dofmap.
                auto entity_verts
                    = mesh::get_entity_vertices(topology->cell_type(), dim)
                          .links(local);
                for (auto lv : entity_verts)
                    points.push_back(x_dofmap(c, static_cast<std::size_t>(lv)));
            }

            std::array<T, 6> b;
            for (int k = 0; k < 3; ++k) {
                b[k] = xg[3 * points[0] + k];
                b[k + 3] = xg[3 * points[0] + k];
            }
            for (auto p : points)
                for (int k = 0; k < 3; ++k) {
                    b[k] = std::min(b[k], xg[3 * p + k]);
                    b[k + 3] = std::max(b[k + 3], xg[3 * p + k]);
                }
            return b;
        }

        /// Union box of a span of leaf boxes (flat {lower, upper}).
        template <std::floating_point T>
        std::array<T, 6> compute_bbox_of_bboxes(
            std::span<const std::pair<std::array<T, 6>, std::int32_t>> leaves)
        {
            std::array<T, 6> b = leaves.front().first;
            for (const auto& [box, _] : leaves)
                for (int k = 0; k < 6; ++k)
                    b[k] = k < 3 ? std::min(b[k], box[k])
                                 : std::max(b[k], box[k]);
            return b;
        }

        /// Recursively build a tree over `leaf_bboxes`; appends nodes to
        /// `bboxes`/`bbox_coordinates` and returns the root node index.
        template <std::floating_point T>
        std::int32_t build_from_leaf(
            std::span<std::pair<std::array<T, 6>, std::int32_t>> leaves,
            std::vector<std::int32_t>& bboxes,
            std::vector<T>& bbox_coordinates)
        {
            if (leaves.size() == 1) {
                const auto [b, entity] = leaves.front();
                bboxes.push_back(entity);
                bboxes.push_back(entity);
                bbox_coordinates.insert(bbox_coordinates.end(), b.begin(), b.end());
                return static_cast<std::int32_t>(bboxes.size() / 2 - 1);
            }

            const std::array<T, 6> b = compute_bbox_of_bboxes<T>(leaves);
            // Split along the longest axis by the midpoint of each box.
            std::array<T, 3> b_diff;
            for (int k = 0; k < 3; ++k)
                b_diff[k] = b[k + 3] - b[k];
            const std::size_t axis = static_cast<std::size_t>(
                std::max_element(b_diff.begin(), b_diff.end()) - b_diff.begin());

            auto middle = std::next(leaves.begin(),
                static_cast<std::ptrdiff_t>(leaves.size() / 2));
            std::nth_element(leaves.begin(), middle, leaves.end(),
                [axis](const auto& p0, const auto& p1) {
                    return p0.first[axis] + p0.first[3 + axis]
                        < p1.first[axis] + p1.first[3 + axis];
                });

            const std::size_t part = leaves.size() / 2;
            const std::int32_t bbox0 = build_from_leaf<T>(
                leaves.first(part), bboxes, bbox_coordinates);
            const std::int32_t bbox1 = build_from_leaf<T>(
                leaves.last(leaves.size() - part), bboxes, bbox_coordinates);

            bboxes.push_back(bbox0);
            bboxes.push_back(bbox1);
            bbox_coordinates.insert(bbox_coordinates.end(), b.begin(), b.end());
            return static_cast<std::int32_t>(bboxes.size() / 2 - 1);
        }

        /// Build a point-cloud tree over `{point, id}` pairs (leaves are
        /// points with lower == upper == point).
        template <std::floating_point T>
        std::int32_t build_from_point(
            std::span<std::pair<std::array<T, 3>, std::int32_t>> points,
            std::vector<std::int32_t>& bboxes,
            std::vector<T>& bbox_coordinates)
        {
            if (points.size() == 1) {
                const std::int32_t id = points[0].second;
                bboxes.push_back(id);
                bboxes.push_back(id);
                bbox_coordinates.insert(bbox_coordinates.end(),
                    points[0].first.begin(), points[0].first.end());
                bbox_coordinates.insert(bbox_coordinates.end(),
                    points[0].first.begin(), points[0].first.end());
                return static_cast<std::int32_t>(bboxes.size() / 2 - 1);
            }

            auto [min, max] = std::ranges::minmax_element(points);
            const std::array<T, 3> b0 = min->first, b1 = max->first;
            std::array<T, 3> b_diff;
            for (int k = 0; k < 3; ++k)
                b_diff[k] = b1[k] - b0[k];
            const std::size_t axis = static_cast<std::size_t>(
                std::max_element(b_diff.begin(), b_diff.end()) - b_diff.begin());

            auto middle = std::next(points.begin(),
                static_cast<std::ptrdiff_t>(points.size() / 2));
            std::nth_element(points.begin(), middle, points.end(),
                [axis](const auto& p0, const auto& p1) {
                    return p0.first[axis] < p1.first[axis];
                });

            const std::size_t part = points.size() / 2;
            const std::int32_t bbox0 = build_from_point<T>(
                points.first(part), bboxes, bbox_coordinates);
            const std::int32_t bbox1 = build_from_point<T>(
                points.last(points.size() - part), bboxes, bbox_coordinates);

            bboxes.push_back(bbox0);
            bboxes.push_back(bbox1);
            bbox_coordinates.insert(bbox_coordinates.end(), b0.begin(), b0.end());
            bbox_coordinates.insert(bbox_coordinates.end(), b1.begin(), b1.end());
            return static_cast<std::int32_t>(bboxes.size() / 2 - 1);
        }

    } // namespace impl_bb

    /// Axis-aligned bounding box tree over mesh entities or a point
    /// cloud, used to accelerate collision and closest-entity queries.
    template <std::floating_point T>
    class BoundingBoxTree {
    public:
        /// Build a tree over mesh entities of topological dimension
        /// `tdim`, optionally restricted to a subset.
        /// @param[in] mesh The mesh.
        /// @param[in] tdim Topological dimension of the entities.
        /// @param[in] padding Padding added to each entity box.
        /// @param[in] entities Entity subset (all local entities when
        /// nullopt).
        BoundingBoxTree(const mesh::Mesh<T>& mesh, int tdim, double padding,
            std::optional<std::span<const std::int32_t>> entities = std::nullopt)
        {
            _tdim = tdim;
            auto* topology = mesh.topology_mutable().get();
            topology->create_entities(tdim);

            // Full entity range unless a subset was given.
            std::vector<std::int32_t> all;
            if (!entities) {
                const std::int32_t num_entities
                    = topology->index_map(tdim)->size_local()
                    + topology->index_map(tdim)->num_ghosts();
                all.resize(static_cast<std::size_t>(num_entities));
                for (std::int32_t e = 0; e < num_entities; ++e)
                    all[static_cast<std::size_t>(e)] = e;
                entities = std::span<const std::int32_t>(all);
            }

            std::vector<std::pair<std::array<T, 6>, std::int32_t>> leaves;
            leaves.reserve(entities->size());
            for (std::int32_t e : *entities) {
                std::array<T, 6> b = impl_bb::compute_bbox_of_entity<T>(mesh, tdim, e);
                for (int k = 0; k < 3; ++k) {
                    b[k] -= static_cast<T>(padding);
                    b[k + 3] += static_cast<T>(padding);
                }
                leaves.emplace_back(b, e);
            }

            if (!leaves.empty())
                impl_bb::build_from_leaf<T>(
                    std::span(leaves), _bboxes, _bbox_coordinates);
        }

        /// Build a point-cloud tree over `{point, id}` pairs.
        explicit BoundingBoxTree(
            std::vector<std::pair<std::array<T, 3>, std::int32_t>> points)
            : _tdim(0)
        {
            if (!points.empty())
                impl_bb::build_from_point<T>(
                    std::span(points), _bboxes, _bbox_coordinates);
        }

        /// Number of nodes.
        std::int32_t num_bboxes() const
        {
            return static_cast<std::int32_t>(_bboxes.size() / 2);
        }

        /// The 6 box coordinates `{lower[3], upper[3]}` of a node.
        std::array<T, 6> get_bbox(std::size_t node) const
        {
            std::array<T, 6> b;
            std::copy_n(_bbox_coordinates.begin() + 6 * node, 6, b.begin());
            return b;
        }

        /// Child node indices (or, for a leaf, the entity index twice).
        std::array<std::int32_t, 2> bbox(std::size_t node) const
        {
            return {_bboxes[2 * node], _bboxes[2 * node + 1]};
        }

        /// Raw box coordinates, `(2*num_bboxes, 3)` flattened.
        std::span<const T> bbox_coordinates() const
        {
            return _bbox_coordinates;
        }

        /// Topological dimension of the entities (0 for a point cloud).
        int tdim() const { return _tdim; }

    private:
        // Node children: node i at slots [2i, 2i+1].
        std::vector<std::int32_t> _bboxes;

        // Node box coordinates, 6 per node.
        std::vector<T> _bbox_coordinates;

        // Topological dimension of the entities.
        int _tdim;
    };

} // namespace hellofem::geometry
