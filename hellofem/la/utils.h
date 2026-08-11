// hellofem::la — shared enums and concepts
// SPDX-License-Identifier: MIT

#pragma once

#include <concepts>
#include <cstdint>
#include <span>

namespace hellofem::la {

    /// Vector norm types.
    enum class Norm : std::int8_t {
        l1, ///< Sum of absolute values
        l2, ///< Euclidean norm
        linf, ///< Maximum absolute value
    };

    /// Concept for the matrix accumulate/set callbacks used by the FEM
    /// assembler: `f(rows, cols, values) -> int`.
    template <class U, class T>
    concept MatSet = std::invocable<U, std::span<const std::int32_t>,
        std::span<const std::int32_t>, std::span<const T>>;

} // namespace hellofem::la
