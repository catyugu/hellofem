// hellofem::la — single-process sparsity pattern (CSR)
// SPDX-License-Identifier: MIT

#pragma once

#include "common/IndexMap.h"

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace hellofem::la {

    /// Sparsity pattern data structure used to initialize sparse matrices.
    ///
    /// Non-zero locations are inserted at scalar-block indices in
    /// `[0, index_map(dim)->size_local())`; the block size is carried as
    /// metadata so the matrix layer can expand a block entry into
    /// `bs` physical rows/columns. After `finalize`, the column indices
    /// of each row are sorted in increasing order and duplicate-free.
    class SparsityPattern {
    public:
        /// Create an empty pattern with the given row/column index maps.
        /// @param[in] maps Index maps for [0] rows and [1] columns.
        /// @param[in] bs Block sizes for [0] rows and [1] columns.
        SparsityPattern(
            std::array<std::shared_ptr<const common::IndexMap>, 2> maps,
            std::array<int, 2> bs);

        /// Create a square pattern from a single index map.
        /// @param[in] index_map Index map for both rows and columns.
        /// @param[in] bs Block size.
        SparsityPattern(std::shared_ptr<const common::IndexMap> index_map,
            int bs = 1);

        SparsityPattern(const SparsityPattern& pattern) = delete;

        /// Move constructor
        SparsityPattern(SparsityPattern&& pattern) = default;

        /// Destructor
        ~SparsityPattern() = default;

        /// Move assignment
        SparsityPattern& operator=(SparsityPattern&& pattern) = default;

        /// Insert a non-zero location at a (row, column) pair.
        /// @param[in] row Local row index.
        /// @param[in] col Local column index.
        void insert(std::int32_t row, std::int32_t col);

        /// Insert non-zero locations at the outer product of `rows` and
        /// `cols`, i.e. the entries `(rows[i], cols[j])` for all i, j.
        /// @param[in] rows Local row indices.
        /// @param[in] cols Local column indices.
        void insert(std::span<const std::int32_t> rows,
            std::span<const std::int32_t> cols);

        /// Insert non-zero locations on the diagonal.
        /// @param[in] rows Local row indices (each also a column index).
        void insert_diagonal(std::span<const std::int32_t> rows);

        /// Finalize the pattern, producing the CSR graph.
        void finalize();

        /// Index map for the given dimension (0 = rows, 1 = columns).
        std::shared_ptr<const common::IndexMap>
        index_map(int dim) const;

        /// Block size for the given dimension (0 = rows, 1 = columns).
        int block_size(int dim) const;

        /// Number of non-zero (block) entries.
        std::int64_t num_nonzeros() const;

        /// CSR graph after `finalize`: column indices (0-based) and row
        /// offsets. Row `i` occupies `cols[offsets[i]:offsets[i+1]]`.
        std::pair<std::span<const std::int32_t>, std::span<const std::int64_t>>
        graph() const;

    private:
        // Index maps for rows and columns
        std::array<std::shared_ptr<const common::IndexMap>, 2> _index_maps;

        // Block sizes for rows and columns
        std::array<int, 2> _bs;

        // Cache of unassembled (row, column) entries as parallel arrays
        // (row-major COO)
        std::vector<std::int32_t> _cache_rows;
        std::vector<std::int32_t> _cache_cols;

        // Finalized CSR: column indices and row offsets
        std::vector<std::int32_t> _edges;
        std::vector<std::int64_t> _offsets;
    };

} // namespace hellofem::la
