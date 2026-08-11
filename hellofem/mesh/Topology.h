// hellofem::mesh — mesh topology (entities and connectivity)
// SPDX-License-Identifier: MIT

#pragma once

#include "cell_types.h"
#include "common/IndexMap.h"
#include "graph/AdjacencyList.h"

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace hellofem::mesh {

    /// Requirement on a range of cell indices.
    template <typename R>
    concept CellRange = std::ranges::input_range<R>
        and std::ranges::sized_range<R>
        and std::is_integral_v<
            std::remove_const_t<std::ranges::range_value_t<R>>>;

    /// Topology stores the mesh entities and their connectivity
    /// (incidence relations). A mesh entity is identified as a pair
    /// `(dim, i)`, where `dim` is the topological dimension and `i` the
    /// index of the entity within that dimension. All entities are
    /// defined in terms of vertices; connectivity relates entities of
    /// different dimensions (e.g. the cells attached to a facet).
    class Topology {
    public:
        /// Create a mesh topology.
        ///
        /// @param[in] cell_types Types of cells.
        /// @param[in] vertex_map Index map of the mesh vertices.
        /// @param[in] cell_maps Index maps of the cells, one per cell
        /// type in `cell_types`.
        /// @param[in] cells Cell-to-vertex connectivity for each cell
        /// type in `cell_types`.
        /// @param[in] original_cell_index Original index of each cell
        /// (e.g. from an input file), preserved across renumbering.
        /// @param[in] num_threads Number of threads to use (>= 1).
        Topology(std::vector<CellType> cell_types,
            std::shared_ptr<const common::IndexMap> vertex_map,
            std::vector<std::shared_ptr<const common::IndexMap>> cell_maps,
            std::vector<std::shared_ptr<graph::AdjacencyList<std::int32_t>>>
                cells,
            const std::optional<std::vector<std::vector<std::int64_t>>>&
                original_cell_index
            = std::nullopt,
            int num_threads = 1);

        /// Copy constructor
        Topology(const Topology& topology) = default;
        /// Move constructor
        Topology(Topology&& topology) = default;
        /// Destructor
        ~Topology() = default;

        // Copy assignment (deleted)
        Topology& operator=(const Topology& topology) = delete;
        /// Move assignment
        Topology& operator=(Topology&& topology) = default;

        /// Topological dimension of the mesh.
        int dim() const noexcept;

        /// Entity types in the topology for a given dimension.
        const std::vector<CellType>& entity_types(int dim) const;

        /// Cell type. Only valid for a topology with one cell type.
        CellType cell_type() const;

        /// Cell types in the topology.
        std::vector<CellType> cell_types() const;

        /// Index maps of the entities of dimension `dim`, one per cell
        /// type.
        std::vector<std::shared_ptr<const common::IndexMap>>
        index_maps(int dim) const;

        /// Index map of the entities of dimension `dim`. Throws if there
        /// is more than one entity type in this dimension or if the map
        /// has not been set (call create_entities first).
        std::shared_ptr<const common::IndexMap> index_map(int dim) const;

        /// Connectivity from entities of type `d0` to entities of type
        /// `d1`, where each of `d0`, `d1` is the pair `(dim, entity-type
        /// index)`. Returns nullptr if not yet computed.
        std::shared_ptr<const graph::AdjacencyList<std::int32_t>>
        connectivity(std::array<int, 2> d0, std::array<int, 2> d1) const;

        /// Connectivity from entities of dimension `d0` to entities of
        /// dimension `d1`, assuming a single entity type per dimension.
        /// Returns nullptr if not yet computed.
        std::shared_ptr<const graph::AdjacencyList<std::int32_t>>
        connectivity(int d0, int d1) const;

        /// Cell permutation information, one entry per cell. See
        /// @ref create_entity_permutations for the encoding.
        /// @pre create_entity_permutations must have been called.
        const std::vector<std::uint32_t>& get_cell_permutation_info() const;

        /// Facet permutations, stored flattened as
        /// `data[cell * facets_per_cell + facet]` where `data[..] % 2` is
        /// the number of reflections and `data[..] / 2` the number of
        /// rotations to apply to the facet.
        /// @pre create_entity_permutations must have been called.
        const std::vector<std::uint8_t>& get_facet_permutations() const;

        /// Indices of inter-process facets of facet type `index`. Always
        /// empty in the single-process build.
        const std::vector<std::int32_t>& interprocess_facets(int index) const;

        /// Indices of inter-process facets (single facet type).
        const std::vector<std::int32_t>& interprocess_facets() const;

        /// Create entities of topological dimension `dim` if they do not
        /// already exist.
        ///
        /// @return True if the entities were created, false if they
        /// already existed.
        bool create_entities(int dim, int num_threads = 1);

        /// Create connectivity between entities of dimensions `d0` and
        /// `d1`, creating the entities first if necessary.
        void create_connectivity(int d0, int d1);

        /// Compute entity permutations and reflections.
        void create_entity_permutations(int num_threads = 1);

        /// Original cell index for each cell type.
        std::vector<std::vector<std::int64_t>> original_cell_index;

    private:
        // Entity types in the mesh: _entity_types[d][i] is the i-th
        // entity type of dimension d.
        std::vector<std::vector<CellType>> _entity_types;

        // Index map for each entity type: _index_maps[{d, i}] is the map
        // of the i-th entity type of dimension d.
        std::map<std::array<int, 2>, std::shared_ptr<const common::IndexMap>>
            _index_maps;

        // Connectivity between entity types, keyed by
        // ((d0, i0), (d1, i1)).
        std::map<std::pair<std::array<int, 2>, std::array<int, 2>>,
            std::shared_ptr<graph::AdjacencyList<std::int32_t>>>
            _connectivity;

        // Facet permutations (local facet, cell), flattened:
        // [c0_f0, c0_f1, ..., c1_f0, ...].
        std::vector<std::uint8_t> _facet_permutations;

        // Cell permutation info, one entry per cell.
        std::vector<std::uint32_t> _cell_permutations;

        // Inter-process facets per facet type.
        std::vector<std::vector<std::int32_t>> _interprocess_facets;
    };

    /// Create a topology from cell-to-vertex connectivities given as
    /// global vertex indices.
    ///
    /// @param[in] cell_types Cell types, one per element of `cells`.
    /// @param[in] cells Cell-to-vertex connectivity for each cell type,
    /// using global vertex indices, flattened row-major.
    /// @param[in] original_cell_index Original index of each cell.
    /// @param[in] num_threads Number of threads to use (>= 1).
    Topology create_topology(std::vector<CellType> cell_types,
        std::vector<std::span<const std::int64_t>> cells,
        std::vector<std::span<const std::int64_t>> original_cell_index,
        int num_threads);

    /// Create a topology for a single cell type from cell-to-vertex
    /// connectivities given as global vertex indices.
    Topology create_topology(std::span<const std::int64_t> cells,
        std::span<const std::int64_t> original_cell_index, CellType cell_type,
        int num_threads);

    /// Create a topology for a subset of entities of dimension `dim` of
    /// an existing topology.
    ///
    /// @return The new topology, the map from entities of dimension
    /// `dim` in the new topology to entities in `topology`, and the map
    /// from vertices in the new topology to vertices in `topology`.
    std::tuple<Topology, std::vector<std::int32_t>, std::vector<std::int32_t>>
    create_subtopology(const Topology& topology, int dim,
        std::span<const std::int32_t> entities);

    /// Get the entity indices for entities defined by their vertices.
    ///
    /// @return Index of the i-th entity in `entities`, or -1 if the
    /// entity cannot be found.
    std::vector<std::int32_t>
    entities_to_index(const Topology& topology, int dim,
        std::span<const std::int32_t> entities);

    /// Cell-cell connections for each pair of cell types sharing a
    /// facet of type `facet_type`. Each list contains flattened
    /// `(cell0, local_facet0, cell1, local_facet1)` quadruples.
    /// @pre Facet-cell and cell-facet connectivity must be computed.
    std::vector<std::vector<std::int32_t>>
    compute_mixed_cell_pairs(const Topology& topology, CellType facet_type);

} // namespace hellofem::mesh
