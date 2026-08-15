// hellofem::fem — expression evaluated over mesh entities
// SPDX-License-Identifier: MIT

#pragma once

#include "Constant.h"
#include "Function.h"
#include "kernel.h"
#include "mesh/Mesh.h"

#include <array>
#include <concepts>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace hellofem::fem {

    /// An expression: a pointwise quantity, defined on reference-cell
    /// points, evaluated over a list of mesh entities (cells, or
    /// (cell, local_facet) pairs).
    ///
    /// The expression depends on a set of coefficient functions and
    /// constants and is evaluated by a kernel that writes the value of
    /// the expression at each reference point. It is a data container:
    /// the caller packs the coefficients and runs
    /// `fem::tabulate_expression` to obtain values.
    template <std::floating_point T>
    class Expression {
    public:
        /// Create an expression.
        /// @param[in] coefficients Coefficient functions the expression
        /// depends on.
        /// @param[in] constants Constants the expression depends on.
        /// @param[in] X Reference points (row-major `(num_points, tdim)`).
        /// @param[in] Xshape Shape of `X`.
        /// @param[in] fn Kernel writing the expression values.
        /// @param[in] value_shape Value shape of the expression.
        Expression(
            std::vector<std::shared_ptr<const Function<T>>> coefficients,
            std::vector<std::shared_ptr<const Constant<T>>> constants,
            std::span<const T> X, std::array<std::size_t, 2> Xshape,
            kernel_t<T> fn, std::vector<std::size_t> value_shape)
            : _coefficients(std::move(coefficients)), _constants(std::move(constants)), _fn(std::move(fn)), _value_shape(std::move(value_shape)), _x_ref(X.begin(), X.end()), _x_shape(Xshape)
        {
        }

        /// Coefficient functions.
        std::span<const std::shared_ptr<const Function<T>>>
        coefficients() const
        {
            return _coefficients;
        }

        /// Constants.
        std::span<const std::shared_ptr<const Constant<T>>> constants() const
        {
            return _constants;
        }

        /// Reference points at which the expression is evaluated.
        std::pair<std::span<const T>, std::array<std::size_t, 2>>
        points() const
        {
            return {std::span(_x_ref), _x_shape};
        }

        /// Value shape of the expression.
        std::span<const std::size_t> value_shape() const
        {
            return _value_shape;
        }

        /// Number of scalar components of the expression value.
        std::size_t value_size() const
        {
            std::size_t s = 1;
            for (std::size_t d : _value_shape)
                s *= d;
            return s;
        }

        /// The kernel computing the expression values.
        const kernel_t<T>& kernel() const { return _fn; }

    private:
        // Coefficient functions.
        std::vector<std::shared_ptr<const Function<T>>> _coefficients;

        // Constants.
        std::vector<std::shared_ptr<const Constant<T>>> _constants;

        // Kernel writing expression values.
        kernel_t<T> _fn;

        // Value shape.
        std::vector<std::size_t> _value_shape;

        // Reference points (row-major).
        std::vector<T> _x_ref;
        std::array<std::size_t, 2> _x_shape;
    };

} // namespace hellofem::fem
