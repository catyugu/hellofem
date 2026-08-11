// hellofem::la — linear operator abstraction
// SPDX-License-Identifier: MIT

#pragma once

#include "Vector.h"

#include <concepts>
#include <functional>

namespace hellofem::la {

    /// Linear operator abstracting the application of a matrix (or a
    /// matrix-free operation) to a vector.
    ///
    /// Wraps a callable computing the *accumulating* matvec `y += op(x)`,
    /// which is the convention shared by `MatrixCSR::mult` and the
    /// iterative solvers (they zero `y` first when a fresh result is
    /// needed).
    ///
    /// @tparam T Value type of the vectors the operator acts on.
    template <typename T>
    class LinearOperator {
    public:
        /// Value type
        using value_type = T;

        /// Vector type the operator acts on
        using vector_type = Vector<T>;

        /// Wrapped callable type: `(x, y) -> void` with `y += A x`.
        using function_type
            = std::function<void(const Vector<T>&, Vector<T>&)>;

        /// Default constructor: an unset operator. `mult` throws until an
        /// operator is wrapped.
        LinearOperator() = default;

        /// Wrap any callable `(const Vector<T>& x, Vector<T>& y) -> void`
        /// computing `y += op(x)`.
        /// @param[in] op The operator to wrap.
        template <typename Op>
            requires std::invocable<Op, const Vector<T>&, Vector<T>&>
        explicit LinearOperator(Op op)
            : _mult(std::move(op))
        {
        }

        /// Apply the operator, accumulating `y += A x`.
        /// @param[in] x Input vector.
        /// @param[in,out] y Accumulator vector.
        void mult(const Vector<T>& x, Vector<T>& y) const { _mult(x, y); }

    private:
        // Wrapped operator (empty if unset)
        function_type _mult;
    };

} // namespace hellofem::la
