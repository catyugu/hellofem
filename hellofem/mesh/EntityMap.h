// hellofem::mesh — bidirectional map between entities of two topologies
// SPDX-License-Identifier: MIT

#pragma once

#include "Topology.h"
#include "common/IndexMap.h"

#include <cstdint>
#include <format>
#include <memory>
#include <ranges>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace hellofem::mesh {

    /// A bidirectional map relating entities in one topology to
    /// entities in a sub-topology of it.
    class EntityMap {
    public:
        /// Constructor of a bidirectional map relating entities of
        /// dimension `dim` in `topology` and `sub_topology`.
        ///
        /// @param[in] topology A mesh topology.
        /// @param[in] sub_topology A topology whose entities of
        /// dimension `dim` are a subset of those in `topology`.
        /// @param[in] dim Topological dimension of the entities.
        /// @param[in] sub_topology_to_topology Index in `topology` of
        /// each entity in `sub_topology`.
        template <typename U>
            requires std::is_convertible_v<std::remove_cvref_t<U>,
                         std::vector<std::int32_t>>
        EntityMap(std::shared_ptr<const Topology> topology,
            std::shared_ptr<const Topology> sub_topology, int dim,
            U&& sub_topology_to_topology)
            : _dim(dim), _topology(topology), _sub_topology_to_topology(std::forward<U>(sub_topology_to_topology)), _sub_topology(sub_topology)
        {
            auto e_imap = sub_topology->index_map(_dim);
            if (!e_imap)
                throw std::runtime_error(std::format(
                    "No index map for entities, call Topology::create_entities({}).",
                    _dim));

            const std::size_t num_ents
                = e_imap->size_local() + e_imap->num_ghosts();
            if (num_ents != _sub_topology_to_topology.size())
                throw std::runtime_error(
                    "Size mismatch between `sub_topology_to_topology` and "
                    "index map.");
        }

        /// Copy constructor
        EntityMap(const EntityMap& map) = default;
        /// Move constructor
        EntityMap(EntityMap&& map) = default;
        /// Destructor
        ~EntityMap() = default;
        /// Copy assignment
        EntityMap& operator=(const EntityMap& map) = default;
        /// Move assignment
        EntityMap& operator=(EntityMap&& map) = default;

        /// Topological dimension of the entities related by this map.
        std::size_t dim() const { return _dim; }

        /// Parent topology.
        std::shared_ptr<const Topology> topology() const { return _topology; }

        /// Sub-topology.
        std::shared_ptr<const Topology> sub_topology() const
        {
            return _sub_topology;
        }

        /// Map entities between the sub-topology and the parent
        /// topology.
        ///
        /// If `inverse` is false, maps `dim`-dimensional entities from
        /// `sub_topology` to `topology`; if true, the reverse. Entities
        /// that do not exist in the target topology are marked -1.
        ///
        /// @note The inverse map is recomputed on every call (not
        /// cached), which is expensive when called repeatedly.
        std::vector<std::int32_t> sub_topology_to_topology(
            CellRange auto&& entities, bool inverse) const
        {
            if (!inverse) {
                auto mapped = std::forward<decltype(entities)>(entities)
                    | std::views::transform([this](std::int32_t i) {
                          return _sub_topology_to_topology[i];
                      });
                return std::vector<std::int32_t>(mapped.begin(), mapped.end());
            }
            else {
                std::unordered_map<std::int32_t, std::int32_t>
                    topology_to_sub_topology;
                topology_to_sub_topology.reserve(
                    _sub_topology_to_topology.size());
                for (std::size_t i = 0; i < _sub_topology_to_topology.size();
                    ++i) {
                    topology_to_sub_topology.insert(
                        {_sub_topology_to_topology[i], static_cast<std::int32_t>(i)});
                }

                auto mapped = std::forward<decltype(entities)>(entities)
                    | std::views::transform(
                        [&topology_to_sub_topology](std::int32_t i) {
                            auto it = topology_to_sub_topology.find(i);
                            return (it != topology_to_sub_topology.end())
                                ? it->second
                                : -1;
                        });
                return std::vector<std::int32_t>(mapped.begin(), mapped.end());
            }
        }

    private:
        // Dimension of the entities.
        std::size_t _dim;

        // Parent topology.
        std::shared_ptr<const Topology> _topology;

        // `_sub_topology_to_topology[i]` is the index in `_topology` of
        // the i-th entity in `_sub_topology`.
        std::vector<std::int32_t> _sub_topology_to_topology;

        // Sub-topology (subset of the entities of `_topology`).
        std::shared_ptr<const Topology> _sub_topology;
    };

} // namespace hellofem::mesh
