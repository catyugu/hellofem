// hellofem — radix sort and permutation helpers
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <bit>
#include <cassert>
#include <concepts>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <optional>
#include <ranges>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace hellofem {

    /// Projection that maps a signed integer to an unsigned integer of the
    /// same width while preserving relative order, by flipping the sign bit.
    struct _unsigned_projection {
        template <std::signed_integral T>
        constexpr std::make_unsigned_t<T> operator()(T e) const noexcept
        {
            using uT = std::make_unsigned_t<T>;
            static_assert(std::numeric_limits<uT>::digits
                == std::numeric_limits<T>::digits + 1);
            return std::bit_cast<uT>(e) ^ (uT(1) << (sizeof(T) * 8 - 1));
        }
    };

    inline constexpr _unsigned_projection unsigned_projection {};

    /// LSD radix sort over a random-access range, sorting by `proj(element)`.
    /// `BITS` is the number of bits handled per pass (bucket count 2^BITS).
    /// Stable: elements equal under `proj` keep their relative order.
    template <int BITS = 8, typename P = std::identity,
        std::ranges::random_access_range R>
    constexpr void radix_sort(R&& range, P proj = {})
    {
        using T = std::ranges::range_value_t<R>;
        using I = std::remove_cvref_t<std::invoke_result_t<P, T>>;
        using uI = std::make_unsigned_t<I>;

        // Signed keys are re-sorted through the order-preserving unsigned
        // projection so the bit tricks below are well-defined.
        if constexpr (std::signed_integral<I>) {
            radix_sort<BITS>(std::forward<R>(range), [&proj](const T& e) -> uI {
                return unsigned_projection(proj(e));
            });
            return;
        }

        if (range.size() <= 1)
            return;

        constexpr uI bucket_size = uI(1) << BITS;
        uI mask = (uI(1) << BITS) - 1;
        constexpr uI top_bit = uI(1) << (sizeof(uI) * 8 - 1);

        // First pass histogram doubles as the pass-0 bucket counters, so the
        // leading pass needs no extra traversal.
        std::array<uI, bucket_size> counter {};
        std::array<uI, bucket_size> offset;

        uI max_value = 0;
        bool all_first_bit = true;
        for (const auto& e : range) {
            uI v = proj(e);
            max_value = std::max(max_value, v);
            all_first_bit = all_first_bit && (v & top_bit);
            counter[v & mask]++;
        }

        // A set top bit on every element carries no ordering information.
        if (all_first_bit)
            max_value = max_value & ~top_bit;

        // Number of BITS-wide passes needed for the largest key.
        uI its = 0;
        while (max_value) {
            max_value >>= BITS;
            its++;
        }

        uI mask_offset = 0;
        std::vector<T> buffer(range.size());
        std::span<T> current = range;
        std::span<T> next = buffer;
        for (uI i = 0; i < its; i++) {
            if (i > 0) {
                std::ranges::fill(counter, 0);
                for (auto c : current)
                    counter[(proj(c) & mask) >> mask_offset]++;
            }

            // Exclusive prefix sum gives the insertion cursor per bucket.
            std::exclusive_scan(counter.begin(), counter.end(), offset.begin(),
                uI(0));
            for (auto c : current) {
                uI bucket = (proj(c) & mask) >> mask_offset;
                next[offset[bucket]++] = c;
            }

            mask = mask << BITS;
            mask_offset += BITS;
            std::swap(current, next);
        }

        if (its % 2 != 0)
            std::ranges::copy(buffer, range.begin());
    }

    /// Permutation array that sorts the rows of a row-major flattened 2D
    /// array `x` (shape0 rows x shape1 cols) by the leading `ncols` columns
    /// (column 0 most significant). Stable.
    template <typename T, int BITS = 16>
    std::vector<std::int32_t> sort_by_perm(std::span<const T> x,
        std::size_t shape1,
        std::optional<std::size_t> ncols
        = std::nullopt)
    {
        static_assert(std::is_integral_v<T>, "Integral required.");

        if (x.empty())
            return {};

        assert(shape1 > 0);
        assert(x.size() % shape1 == 0);
        std::size_t n = ncols.value_or(shape1);
        assert(n <= shape1);
        const std::size_t shape0 = x.size() / shape1;
        std::vector<std::int32_t> perm(shape0);
        std::iota(perm.begin(), perm.end(), 0);

        // Sort columns right to left (least significant digit first).
        std::vector<T> column(shape0);
        for (std::size_t i = 0; i < n; ++i) {
            std::size_t col = n - 1 - i;
            for (std::size_t j = 0; j < shape0; ++j)
                column[j] = x[j * shape1 + col];
            radix_sort<BITS>(perm, [column = std::cref(column)](auto index) {
                return column.get()[index];
            });
        }

        return perm;
    }

    /// Same as the overload above, but takes columns as individual contiguous
    /// spans, most significant first.
    template <typename T, int BITS = 16>
    std::vector<std::int32_t>
    sort_by_perm(std::span<std::span<const T>> x)
    {
        static_assert(std::is_integral_v<T>, "Integral required.");
        if (x.empty())
            return {};

        std::size_t ncols = x.size();
        std::size_t nrows = x.front().size();
        std::vector<std::int32_t> perm(nrows);
        std::iota(perm.begin(), perm.end(), 0);

        for (std::size_t i = 0; i < ncols; ++i) {
            std::size_t col = ncols - 1 - i;
            assert(x[col].size() == nrows);
            radix_sort<BITS>(perm, [column = x[col]](auto index) {
                return column[index];
            });
        }
        return perm;
    }

} // namespace hellofem
