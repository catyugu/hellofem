// hellofem::common — partition a global range into contiguous chunks
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cassert>
#include <cstdint>

namespace hellofem::common {

    /// Split the global range [0, N) into `size` contiguous, nearly equal
    /// parts and return the part with the given `index`. Part `i` ends where
    /// part `i + 1` begins.
    constexpr std::array<std::int64_t, 2> local_range(int index, std::int64_t N,
        int size)
    {
        assert(index >= 0);
        assert(N >= 0);
        assert(size > 0);

        const std::int64_t n = N / size;
        const std::int64_t r = N % size;

        if (index < r)
            return {index * (n + 1), index * (n + 1) + n + 1};
        else
            return {index * n + r, index * n + r + n};
    }

} // namespace hellofem::common
