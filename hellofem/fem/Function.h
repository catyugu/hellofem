// hellofem::fem — finite element function
// SPDX-License-Identifier: MIT

#pragma once

#include "FunctionSpace.h"
#include "la/Vector.h"

#include <array>
#include <concepts>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace hellofem::fem {

    /// A function on a function space: a coefficient vector over the
    /// space's dofs.
    ///
    /// The coefficient vector is laid out as `bs * index_map->size_local()`
    /// entries (blocked dofs expanded to physical dofs), matching
    /// `la::Vector`.
    template <std::floating_point T>
    class Function {
    public:
        /// Value type.
        using value_type = T;

        /// Evaluation of a field: given physical coordinates (flattened
        /// `(num_points, gdim)` row-major) return field values
        /// (flattened `(num_points, value_size)` row-major).
        using eval_fn = std::function<std::pair<std::vector<T>,
            std::array<std::size_t, 2>>(std::span<const T>,
            std::array<std::size_t, 2>)>;

        /// Create a zero function on a function space, allocating its
        /// coefficient vector.
        /// @param[in] V The function space.
        explicit Function(std::shared_ptr<const FunctionSpace<T>> V)
            : _V(std::move(V)), _x(std::make_shared<la::Vector<T>>(_V->dofmap()->index_map, _V->dofmap()->index_map_bs()))
        {
        }

        /// The function space.
        std::shared_ptr<const FunctionSpace<T>> function_space() const
        {
            return _V;
        }

        /// The coefficient vector (shared, so sub-functions can share).
        std::shared_ptr<la::Vector<T>> x() const { return _x; }

        /// Interpolate a field into the function by evaluating it at the
        /// element's interpolation points on every cell and applying the
        /// element's interpolation operator.
        /// @param[in] f Field evaluator (physical coordinates -> values).
        void interpolate(const eval_fn& f);

        /// Interpolate a function on the same space (copy of its
        /// coefficients).
        void interpolate(const Function<T>& u);

        /// Evaluate the function at a set of physical points.
        ///
        /// @param[in] x Point coordinates, flattened row-major with shape
        /// `(num_points, gdim)`.
        /// @param[in] xshape Shape of `x`.
        /// @param[in] cells For each point, the index of the cell that
        /// contains it (negative entries are skipped).
        /// @param[in,out] u Values, flattened row-major with shape
        /// `(num_points, value_size)`, zeroed before accumulation.
        /// @param[in] ushape Shape of `u`.
        /// @param[in] tol Newton tolerance for pull-back (non-affine
        /// geometry only).
        /// @param[in] maxit Newton iteration limit for pull-back.
        void eval(std::span<const T> x, std::array<std::size_t, 2> xshape,
            std::span<const std::int32_t> cells, std::span<T> u,
            std::array<std::size_t, 2> ushape, double tol = 1e-12,
            int maxit = 20) const;

        /// Evaluate the function at arbitrary physical points, locating
        /// the containing cell automatically (single-process).
        ///
        /// @param[in] x Point coordinates, flattened row-major with shape
        /// `(num_points, gdim)`.
        /// @param[in] xshape Shape of `x`.
        /// @return Values, flattened row-major with shape
        /// `(num_points, value_size)`.
        std::pair<std::vector<T>, std::array<std::size_t, 2>>
        eval(std::span<const T> x, std::array<std::size_t, 2> xshape) const;

        /// Name.
        std::string name = "u";

    private:
        // Function space.
        std::shared_ptr<const FunctionSpace<T>> _V;

        // Coefficient vector.
        std::shared_ptr<la::Vector<T>> _x;
    };

} // namespace hellofem::fem
