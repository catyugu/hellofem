// hellofem::la — single-process sparsity pattern (CSR)
// SPDX-License-Identifier: MIT

#include "SparsityPattern.h"

#include <algorithm>
#include <cassert>
#include <numeric>
#include <ranges>
#include <stdexcept>
#include <utility>

namespace hellofem::la {

    /// Bucket (row, column) entries by row (not sorted or deduped within
    /// a row).
    ///
    /// @return Row offsets (size `num_rows + 1`) and column indices
    /// grouped by row: row `i` occupies `cols[offsets[i]:offsets[i+1]]`.
    static std::pair<std::vector<std::int64_t>, std::vector<std::int32_t>>
    bucket_by_row(std::span<const std::int32_t> rows,
        std::span<const std::int32_t> cols, std::int32_t num_rows)
    {
        assert(rows.size() == cols.size());
        std::vector<std::int64_t> offsets(num_rows + 1, 0);
        for (std::int32_t row : rows)
            ++offsets[row + 1];
        std::partial_sum(offsets.begin(), offsets.end(), offsets.begin());

        std::vector<std::int64_t> pos(offsets.begin(), std::prev(offsets.end()));
        std::vector<std::int32_t> bucketed(cols.size());
        for (std::size_t i = 0; i < rows.size(); ++i)
            bucketed[pos[rows[i]]++] = cols[i];

        return {std::move(offsets), std::move(bucketed)};
    }

    SparsityPattern::SparsityPattern(
        std::array<std::shared_ptr<const common::IndexMap>, 2> maps,
        std::array<int, 2> bs)
        : _index_maps(std::move(maps)), _bs(bs)
    {
        assert(_index_maps[0]);
    }

    SparsityPattern::SparsityPattern(
        std::shared_ptr<const common::IndexMap> index_map, int bs)
        : _bs({bs, bs})
    {
        assert(index_map);
        _index_maps[0] = index_map;
        _index_maps[1] = std::move(index_map);
    }

    void SparsityPattern::insert(std::int32_t row, std::int32_t col)
    {
        if (!_offsets.empty())
            throw std::runtime_error("Cannot insert into sparsity pattern. It "
                                     "has already been finalized");
        assert(_index_maps[0]);
        _cache_rows.push_back(row);
        _cache_cols.push_back(col);
    }

    void SparsityPattern::insert(std::span<const std::int32_t> rows,
        std::span<const std::int32_t> cols)
    {
        if (!_offsets.empty())
            throw std::runtime_error("Cannot insert into sparsity pattern. It "
                                     "has already been finalized");
        assert(_index_maps[0]);
        for (std::int32_t row : rows) {
            _cache_rows.insert(_cache_rows.end(), cols.size(), row);
            _cache_cols.insert(_cache_cols.end(), cols.begin(), cols.end());
        }
    }

    void SparsityPattern::insert_diagonal(std::span<const std::int32_t> rows)
    {
        if (!_offsets.empty())
            throw std::runtime_error("Cannot insert into sparsity pattern. It "
                                     "has already been finalized");
        assert(_index_maps[0]);
        _cache_rows.insert(_cache_rows.end(), rows.begin(), rows.end());
        _cache_cols.insert(_cache_cols.end(), rows.begin(), rows.end());
    }

    std::shared_ptr<const common::IndexMap>
    SparsityPattern::index_map(int dim) const
    {
        return _index_maps.at(dim);
    }

    int SparsityPattern::block_size(int dim) const { return _bs[dim]; }

    void SparsityPattern::finalize()
    {
        if (!_offsets.empty())
            throw std::runtime_error("Sparsity pattern has already been "
                                     "finalized.");

        assert(_index_maps[0]);
        const std::int32_t num_rows = _index_maps[0]->size_local();

        // Bucket the insertion cache by row, then drop the cache early to
        // reduce peak memory for the rest of finalize().
        const auto [cache_offsets, cache_cols]
            = bucket_by_row(_cache_rows, _cache_cols, num_rows);
        std::vector<std::int32_t>().swap(_cache_rows);
        std::vector<std::int32_t>().swap(_cache_cols);

        // De-duplicate each row's raw column list with a generation-
        // stamped marker: `last_seen[col] == i` means col has already
        // been recorded for row i. Rows are visited in increasing order,
        // so the row index itself is the stamp -- no reset between rows.
        const std::int32_t num_cols = _index_maps[1]->size_local();
        std::vector<std::int32_t> last_seen(num_cols, -1);

        _edges.reserve(cache_cols.size());
        _offsets.reserve(num_rows + 1);
        _offsets.push_back(0);
        std::vector<std::int32_t> row;
        for (std::int32_t i = 0; i < num_rows; ++i) {
            row.clear();
            for (std::int64_t k = cache_offsets[i]; k < cache_offsets[i + 1];
                ++k) {
                if (std::int32_t c = cache_cols[k]; last_seen[c] != i) {
                    last_seen[c] = i;
                    row.push_back(c);
                }
            }
            std::ranges::sort(row);
            _edges.insert(_edges.end(), row.begin(), row.end());
            _offsets.push_back(_offsets.back() + row.size());
        }

        _edges.shrink_to_fit();
        _offsets.shrink_to_fit();
    }

    std::int64_t SparsityPattern::num_nonzeros() const
    {
        if (_offsets.empty())
            throw std::runtime_error("Sparsity pattern has not been finalized.");
        return _edges.size();
    }

    std::pair<std::span<const std::int32_t>, std::span<const std::int64_t>>
    SparsityPattern::graph() const
    {
        if (_offsets.empty())
            throw std::runtime_error("Sparsity pattern has not been finalized.");
        return {_edges, _offsets};
    }

} // namespace hellofem::la
