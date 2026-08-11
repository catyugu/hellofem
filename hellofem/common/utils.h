// hellofem::common — generic small utilities
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstddef>
#include <functional>
#include <iterator>
#include <ranges>
#include <stdexcept>
#include <utility>
#include <vector>

namespace hellofem::common {

    /// Sort two arrays together by the values in `indices`. Duplicate
    /// indices are removed, keeping the entry with the smallest value.
    template <std::ranges::input_range U, std::ranges::input_range V>
    std::pair<std::vector<typename U::value_type>,
        std::vector<typename V::value_type>>
    sort_unique(const U& indices, const V& values)
    {
        if (indices.size() != values.size())
            throw std::runtime_error(
                "Cannot sort two arrays of different lengths");

        using T = std::pair<typename U::value_type, typename V::value_type>;
        std::vector<T> data(indices.size());
        std::ranges::transform(indices, values, data.begin(),
            [](const auto& idx, const auto& v) -> T {
                return {idx, v};
            });

        std::ranges::sort(data);
        auto it = std::ranges::unique(data, [](const auto& a, const auto& b) {
            return a.first == b.first;
        }).begin();

        std::vector<typename U::value_type> indices_new;
        std::vector<typename V::value_type> values_new;
        std::size_t n = std::ranges::distance(data.begin(), it);
        indices_new.reserve(n);
        values_new.reserve(n);
        std::transform(data.begin(), it, std::back_inserter(indices_new),
            [](const auto& d) { return d.first; });
        std::transform(data.begin(), it, std::back_inserter(values_new),
            [](const auto& d) { return d.second; });

        return {std::move(indices_new), std::move(values_new)};
    }

    /// Local hash of an object (single-process flavour of a distributed
    /// hash; no MPI gather/broadcast).
    template <class T>
    std::size_t hash_local(const T& x)
    {
        return std::hash<T> {}(x);
    }

} // namespace hellofem::common
