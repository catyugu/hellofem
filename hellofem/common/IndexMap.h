// hellofem::common — single-process index map
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace hellofem::common {

    class IndexMap;

    /// Enum to control preservation of ghost index ordering in sub-IndexMaps.
    /// Retained for API parity; single-process maps never carry ghosts.
    enum class IndexMapOrder : bool {
        preserve = true,
        any = false
    };

    /// Create a new index map from a subset of indices in an existing map.
    ///
    /// @param[in] imap Parent map to draw the subset from.
    /// @param[in] indices Local indices in `imap` (owned) to include in the
    /// new map. Must be sorted and duplicate-free.
    /// @param[in] order Unused in the single-process build (ghosts do not
    /// exist); kept for signature parity.
    /// @return The (i) new index map, whose global numbering is a contiguous
    /// renumbering of the selected indices, and (ii) a map from local indices
    /// in the submap to local indices in the original map.
    std::pair<IndexMap, std::vector<std::int32_t>> create_sub_index_map(
        const IndexMap& imap, std::span<const std::int32_t> indices,
        IndexMapOrder order = IndexMapOrder::any);

    /// Distributed index map reduced to a single process.
    ///
    /// The map owns a contiguous block of global indices
    /// `[offset, offset + local_size)`. Since there is exactly one process,
    /// no ghost indices, ownership or communication ever arise; the ghost-
    /// related API is retained (degenerate: always empty) so downstream
    /// topology/dofmap code keeps its natural shape.
    class IndexMap {
    public:
        /// Create a map owning `local_size` indices starting at global
        /// `offset`.
        IndexMap(std::int64_t offset, std::int32_t local_size);

        /// Move constructor
        IndexMap(IndexMap&& map) = default;

        /// Destructor
        ~IndexMap() = default;

        /// Move assignment
        IndexMap& operator=(IndexMap&& map) = default;

        // Copy is disallowed: maps are heavyweight index holders.
        IndexMap(const IndexMap& map) = delete;
        IndexMap& operator=(const IndexMap& map) = delete;

        /// Range of global indices owned by this process.
        std::array<std::int64_t, 2> local_range() const noexcept;

        /// Number of ghost indices (always 0 in the single-process build).
        std::int32_t num_ghosts() const noexcept;

        /// Number of indices owned by this process.
        std::int32_t size_local() const noexcept;

        /// Number of indices across the (single) process.
        std::int64_t size_global() const noexcept;

        /// Local-to-global map for ghosts (always empty here).
        std::span<const std::int64_t> ghosts() const noexcept;

        /// Compute global indices for an array of local indices.
        void local_to_global(std::span<const std::int32_t> local,
                             std::span<std::int64_t> global) const;

        /// Compute local indices for an array of global indices. Returns -1
        /// for global indices not owned by this process.
        void global_to_local(std::span<const std::int64_t> global,
                             std::span<std::int32_t> local) const;

        /// Global index for every local index (0, 1, 2, ...).
        std::vector<std::int64_t> global_indices() const;

        /// Ranks owning each ghost index (always empty here).
        std::span<const int> owners() const noexcept { return _owners; }

        /// (Ranks, local indices) of owned indices ghosted elsewhere.
        /// Single-process: empty.
        std::pair<std::vector<int>, std::vector<std::int32_t>>
        index_to_dest_ranks() const;

        /// Local index of owned indices that are ghosts elsewhere. Always
        /// empty in the single-process build.
        std::vector<std::int32_t> shared_indices() const;

        /// Ranks that own this process's ghosts (always empty).
        std::span<const int> src() const noexcept { return _src; }

        /// Ranks that ghost this process's indices (always empty).
        std::span<const int> dest() const noexcept { return _dest; }

    private:
        // Global index block owned by this (the only) process
        std::array<std::int64_t, 2> _local_range;

        // Number of indices across all (one) processes
        std::int64_t _size_global;

        // Local-to-global map for ghost indices (always empty)
        std::vector<std::int64_t> _ghosts;

        // Owning rank of each ghost index (always empty)
        std::vector<int> _owners;

        // Ranks that own ghosts (always empty)
        std::vector<int> _src;

        // Ranks that ghost owned indices (always empty)
        std::vector<int> _dest;
    };

} // namespace hellofem::common
