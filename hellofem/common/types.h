// hellofem::common — scalar concepts, mdspan aliases and numeric traits
// SPDX-License-Identifier: MIT

#pragma once

#include "../basis/mdspan.hpp"

#include <complex>
#include <concepts>
#include <cstddef>
#include <type_traits>

// Vendor mdspan header defines its types in MDSPAN_IMPL_STANDARD_NAMESPACE
// (std) and MDSPAN_IMPL_PROPOSED_NAMESPACE (experimental). Surface the
// `md` / `mdex` aliases at the hellofem root so every module can reach them
// through enclosing-namespace lookup.
namespace hellofem {
    // NOLINTBEGIN
    namespace md = MDSPAN_IMPL_STANDARD_NAMESPACE;
    namespace mdex
        = MDSPAN_IMPL_STANDARD_NAMESPACE::MDSPAN_IMPL_PROPOSED_NAMESPACE;
    // NOLINTEND
    /// A scalar is a floating-point type or a complex type over one.
    template <class T>
    struct is_custom_scalar : std::false_type { };

    template <class T>
    concept scalar = std::floating_point<T>
        || std::is_same_v<T, std::complex<typename T::value_type>>
        || is_custom_scalar<T>::value;

    /// Underlying value type of a scalar (unwraps std::complex).
    template <scalar T, typename = void>
    struct scalar_value {
        typedef T type;
    };

    template <scalar T>
    struct scalar_value<T, std::void_t<typename T::value_type>> {
        typedef typename T::value_type type;
    };

    template <scalar T>
    using scalar_value_t = typename scalar_value<T>::type;

    /// Rank-2 mdspan-like concept: exposes value_type, static rank() == 2,
    /// two-index access and per-dimension extent queries.
    template <typename T>
    concept MDSpanRank2 = requires(std::remove_cvref_t<T> x, std::size_t i) {
        typename std::remove_cvref_t<T>::value_type;
        requires std::remove_cvref_t<T>::rank() == 2;
        x(i, i);
        { x.extent(i) } -> std::integral;
    };
} // namespace hellofem

namespace hellofem::common {
} // namespace hellofem::common
