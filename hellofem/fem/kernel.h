// hellofem::fem — integral kernel types
// SPDX-License-Identifier: MIT

#pragma once

#include <concepts>
#include <cstdint>
#include <functional>
#include <type_traits>

namespace hellofem::fem {

    /// Type of integral in a form.
    enum class IntegralType : std::int8_t {
        cell = 0, ///< Cell integral
        exterior_facet = 1, ///< Exterior facet integral
        interior_facet = 2 ///< Interior facet integral
    };

    /// Scalar type concept: floating point (complex deferred).
    template <typename T>
    concept scalar = std::floating_point<T>;

    /// Geometry (floating point) type for a scalar type.
    template <scalar T>
    using geometry_t = double;

    /// A kernel computes the element tensor of an integral over a cell
    /// (or facet). It receives pre-packaged, per-cell data and writes
    /// the element tensor to `Ae`.
    ///
    /// @param[out] Ae Element tensor. For a bilinear form the shape is
    /// `(ndofs0*bs0, ndofs1*bs1)` row-major; for a linear form `(ndofs*bs)`.
    /// @param[in] coeffs Packed coefficient dof values for this cell,
    /// `[offset + i]` where `i` indexes the coefficient's dofs.
    /// @param[in] constants Concatenated constant values.
    /// @param[in] cdofs This cell's coordinate dofs, `3 * num_geom_dofs`
    /// entries (geometry points are always 3D).
    /// @param[in] entity Local entity index (facets), else nullptr.
    /// @param[in] perm Facet permutation, else nullptr.
    /// @param[in] user Opaque user data (unused).
    template <std::floating_point T, std::floating_point U = double>
    using kernel_t = std::function<void(T*, const T*, const T*, const U*,
        const int*, const std::uint8_t*, void*)>;

} // namespace hellofem::fem
