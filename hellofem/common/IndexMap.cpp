// hellofem::common — single-process index map implementation
// SPDX-License-Identifier: MIT

#include "IndexMap.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <numeric>

using namespace hellofem::common;

IndexMap::IndexMap(std::int64_t offset, std::int32_t local_size)
    : _local_range({offset, offset + local_size}),
      _size_global(offset + local_size)
{
    assert(local_size >= 0);
}

std::array<std::int64_t, 2> IndexMap::local_range() const noexcept
{
    return _local_range;
}

std::int32_t IndexMap::num_ghosts() const noexcept
{
    return static_cast<std::int32_t>(_ghosts.size());
}

std::int32_t IndexMap::size_local() const noexcept
{
    return static_cast<std::int32_t>(_local_range[1] - _local_range[0]);
}

std::int64_t IndexMap::size_global() const noexcept { return _size_global; }

std::span<const std::int64_t> IndexMap::ghosts() const noexcept
{
    return _ghosts;
}

void IndexMap::local_to_global(std::span<const std::int32_t> local,
                               std::span<std::int64_t> global) const
{
    assert(local.size() == global.size());
    std::ranges::transform(local, global.begin(), [this](auto i) {
        return _local_range[0] + i;
    });
}

void IndexMap::global_to_local(std::span<const std::int64_t> global,
                               std::span<std::int32_t> local) const
{
    assert(global.size() == local.size());
    for (std::size_t i = 0; i < global.size(); ++i) {
        if (global[i] >= _local_range[0] and global[i] < _local_range[1])
            local[i] = static_cast<std::int32_t>(global[i] - _local_range[0]);
        else
            local[i] = -1;
    }
}

std::vector<std::int64_t> IndexMap::global_indices() const
{
    std::vector<std::int64_t> out(_local_range[1] - _local_range[0]);
    std::iota(out.begin(), out.end(), _local_range[0]);
    return out;
}

std::pair<std::vector<int>, std::vector<std::int32_t>>
IndexMap::index_to_dest_ranks() const
{
    // No other process exists, so nothing is ever ghosted.
    return {};
}

std::vector<std::int32_t> IndexMap::shared_indices() const
{
    return {};
}

std::pair<IndexMap, std::vector<std::int32_t>>
hellofem::common::create_sub_index_map(const IndexMap& imap,
                                       std::span<const std::int32_t> indices,
                                       IndexMapOrder)
{
    // Single-process semantics mirror a one-rank MPI_Exscan: the submap is
    // a contiguous renumbering of the selected owned indices starting at
    // global index 0.
    (void)imap;
    std::int32_t local_size = static_cast<std::int32_t>(indices.size());

    std::vector<std::int32_t> sub_imap_to_imap(indices.begin(), indices.end());
    return {IndexMap(0, local_size), std::move(sub_imap_to_imap)};
}
