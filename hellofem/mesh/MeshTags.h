// hellofem::mesh — tags associated with mesh topology entities
// SPDX-License-Identifier: MIT

#pragma once

#include "Topology.h"
#include "common/utils.h"
#include "graph/AdjacencyList.h"
#include "spdlog/spdlog.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iterator>
#include <memory>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace hellofem::mesh {

    /// MeshTags associates values with mesh topology entities. The
    /// entity index (local to the process) identifies the entity.
    /// MeshTags is *sparse*: tags can be associated with an arbitrary
    /// subset of entities, and each entity carries at most one tag.
    ///
    /// @tparam T Type of the tag value.
    template <typename T>
    class MeshTags {
    public:
        /// Create MeshTags from entities of given dimension on a mesh.
        ///
        /// @param[in] topology Mesh topology on which tags are
        /// associated.
        /// @param[in] dim Topological dimension of tagged entities.
        /// @param[in] indices Local entity indices.
        /// @param[in] values Tag values, one per index.
        /// @param[in] name Name of the meshtags.
        /// @pre `indices` must be sorted and unique.
        template <typename U, typename V>
            requires std::is_convertible_v<std::remove_cvref_t<U>,
                         std::vector<std::int32_t>>
                         and std::is_convertible_v<std::remove_cvref_t<V>,
                             std::vector<T>>
        MeshTags(std::shared_ptr<const Topology> topology, int dim, U&& indices,
            V&& values, std::string name = "mesh_tags")
            : _topology(std::move(topology)), _dim(dim), _indices(std::forward<U>(indices)), _values(std::forward<V>(values)), _name(std::move(name))
        {
            if (_indices.size() != _values.size())
                throw std::runtime_error(
                    "Indices and values arrays must have same size.");
#ifndef NDEBUG
            if (!std::ranges::is_sorted(_indices))
                throw std::runtime_error("MeshTag data is not sorted");
            if (std::adjacent_find(_indices.begin(), _indices.end())
                != _indices.end())
                throw std::runtime_error("MeshTag data has duplicates");
#endif
        }

        /// Copy constructor
        MeshTags(const MeshTags& tags) = default;
        /// Move constructor
        MeshTags(MeshTags&& tags) = default;
        /// Destructor
        ~MeshTags() = default;
        /// Copy assignment
        MeshTags& operator=(const MeshTags& tags) = default;
        /// Move assignment
        MeshTags& operator=(MeshTags&& tags) = default;

        /// Find all entities with a given tag value.
        /// @return Sorted indices of tagged entities.
        std::vector<std::int32_t> find(const T value) const
        {
            std::vector<std::int32_t> indices;
            for (std::size_t i = 0; i < _values.size(); ++i)
                if (_values[i] == value)
                    indices.push_back(_indices[i]);
            return indices;
        }

        /// Indices of tagged topology entities (local to the process).
        /// Sorted.
        std::span<const std::int32_t> indices() const { return _indices; }

        /// Values attached to the tagged entities.
        std::span<const T> values() const { return _values; }

        /// Topological dimension of the tagged entities.
        int dim() const { return _dim; }

        /// Number of tagged entities.
        std::size_t size() const { return _indices.size(); }

        /// Associated topology.
        std::shared_ptr<const Topology> topology() const { return _topology; }

        /// Name.
        const std::string& name() const { return _name; }

        /// Set the name.
        void name(std::string name) { _name = std::move(name); }

    private:
        // Associated topology.
        std::shared_ptr<const Topology> _topology;

        // Topological dimension of the tagged entities.
        int _dim;

        // Local-to-process indices of tagged entities.
        std::vector<std::int32_t> _indices;

        // Values attached to entities.
        std::vector<T> _values;

        // Name.
        std::string _name;
    };

    /// Create MeshTags from entities defined by their vertex indices.
    ///
    /// @param[in] topology Mesh topology that the tags are associated
    /// with.
    /// @param[in] dim Topological dimension of tagged entities.
    /// @param[in] entities Local vertex indices for tagged entities
    /// (one row per tagged entity).
    /// @param[in] values Tag value for each entity in `entities`.
    /// @param[in] name Name of the meshtags.
    /// @note Entities that do not exist are ignored.
    /// @warning `entities` must not contain duplicates.
    template <typename V>
    MeshTags<std::remove_cvref_t<std::ranges::range_value_t<V>>>
    create_meshtags(std::shared_ptr<const Topology> topology, int dim,
        const graph::AdjacencyList<std::int32_t>& entities, V&& values,
        std::string name = "mesh_tags")
    {
        using T = std::remove_cvref_t<std::ranges::range_value_t<V>>;

        spdlog::info(
            "Building MeshTags object from tagged entities (defined by "
            "vertices).");

        // Compute the indices of the topology entities (index is -1 if
        // the entity cannot be found).
        assert(topology);
        const std::vector<std::int32_t> indices
            = entities_to_index(*topology, dim, entities.array());
        if (indices.size() != std::ranges::size(values))
            throw std::runtime_error(
                "Duplicate mesh entities when building MeshTags object.");

        // Sort the indices and values by indices.
        std::vector<T> values_vec(values.begin(), values.end());
        auto [indices_sorted, values_sorted]
            = common::sort_unique(indices, values_vec);

        // Remove entities that were not found (index == -1).
        auto it0 = std::ranges::lower_bound(indices_sorted, 0);
        const std::size_t pos0
            = std::ranges::distance(indices_sorted.begin(), it0);
        indices_sorted.erase(indices_sorted.begin(), it0);
        values_sorted.erase(
            values_sorted.begin(), std::next(values_sorted.begin(), pos0));

        return MeshTags<T>(std::move(topology), dim, std::move(indices_sorted),
            std::move(values_sorted), std::move(name));
    }

} // namespace hellofem::mesh
