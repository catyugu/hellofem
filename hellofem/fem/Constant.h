// hellofem::fem — scalar and vector constant
// SPDX-License-Identifier: MIT

#pragma once

#include <concepts>
#include <cstddef>
#include <span>
#include <vector>

namespace hellofem::fem {

    /// A constant value appearing in a form, e.g. a material parameter.
    ///
    /// A rank-0 constant holds a single value; a rank-1 constant holds a
    /// vector. The value is stored flattened row-major together with its
    /// shape so that forms can inspect the number of components.
    template <std::floating_point T>
    class Constant {
    public:
        /// Value type.
        using value_type = T;

        /// Create a scalar (rank-0) constant.
        explicit Constant(T c) : value({c}), shape() { }

        /// Create a rank-1 constant from a vector of values.
        explicit Constant(std::span<const T> c)
            : value(c.begin(), c.end()), shape({c.size()})
        {
        }

        /// Create a constant with an explicit value shape.
        Constant(std::span<const T> c, std::span<const std::size_t> shape)
            : value(c.begin(), c.end()), shape(shape.begin(), shape.end())
        {
        }

        /// Flattened constant values.
        std::vector<T> value;

        /// Shape of the constant.
        std::vector<std::size_t> shape;
    };

} // namespace hellofem::fem
