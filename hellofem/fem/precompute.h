// hellofem::fem — precomputed quadrature data and cell-kernel factory
// SPDX-License-Identifier: MIT

#pragma once

#include "CoordinateElement.h"
#include "FiniteElement.h"
#include "kernel.h"
#include "mesh/cell_types.h"

#include <concepts>
#include <cstdint>
#include <span>
#include <vector>

namespace hellofem::fem {

    /// Per-cell quadrature data handed to a weak form: the physical
    /// basis, weights, Jacobian determinants and coefficient values at
    /// the quadrature points of the current cell.
    template <std::floating_point T>
    struct CellKernelData {
        /// Test basis values, `(num_points, ndofs0)`.
        std::span<const T> phi0;

        /// Test basis physical gradients, `(num_points, ndofs0, tdim)`.
        std::span<const T> dphi0;

        /// Trial basis values, `(num_points, ndofs1)`.
        std::span<const T> phi1;

        /// Trial basis physical gradients, `(num_points, ndofs1, tdim)`.
        std::span<const T> dphi1;

        /// Quadrature weights, `(num_points)`.
        std::span<const T> w;

        /// Jacobian determinants, `(num_points)`.
        std::span<const T> detJ;

        /// Coefficient values at the quadrature points, `(num_coeffs,
        /// num_points)`, row `c` at `c * num_points`.
        std::span<const T> coeffs;

        /// Concatenated constant values in form order. The weak form
        /// knows its own constant count.
        const T* constants;

        /// Number of quadrature points.
        int num_points;

        /// Number of test dofs per cell.
        int num_dofs0;

        /// Number of trial dofs per cell.
        int num_dofs1;

        /// Topological dimension.
        int tdim;
    };

    /// A weak form: accumulate the element tensor `Ae` from the
    /// quadrature data of a single cell.
    ///
    /// @param[out] Ae Element tensor. For a bilinear form the shape is
    /// `(ndofs0, ndofs1)` row-major; for a linear form `(ndofs0)`.
    /// @param[in] data Per-cell quadrature data.
    template <std::floating_point T>
    using cell_kernel_weak_fn_t = void (*)(T* Ae,
        const CellKernelData<T>& data);

    /// Reference-cell data for assembling cell integrals: the quadrature
    /// rule and the reference basis values of the test, trial and
    /// coefficient elements, and of the coordinate element, at the
    /// quadrature points.
    template <std::floating_point T>
    class PrecomputeData {
    public:
        /// Precompute the data for a cell type.
        /// @param[in] cell_type Cell type.
        /// @param[in] test_element Test function element.
        /// @param[in] trial_element Trial function element.
        /// @param[in] coeff_elements Elements of the coefficients the
        /// integral depends on (may be empty).
        /// @param[in] coord_element Coordinate element.
        /// @param[in] quadrature_degree Degree the rule must integrate
        /// exactly.
        PrecomputeData(mesh::CellType cell_type,
            const FiniteElement<T>& test_element,
            const FiniteElement<T>& trial_element,
            const std::vector<const FiniteElement<T>*>& coeff_elements,
            const CoordinateElement<T>& coord_element, int quadrature_degree);

        /// Reference quadrature points, `(num_points, tdim)`.
        std::span<const T> points() const { return Xq; }

        /// Quadrature weights, `(num_points)`.
        std::span<const T> weights() const { return wq; }

        /// Test basis values, `(num_points, ndofs0)`.
        std::span<const T> test_phi() const { return phi0; }

        /// Test basis reference gradients, `(num_points, ndofs0, tdim)`.
        std::span<const T> test_dphi_ref() const { return dphi0; }

        /// Trial basis values, `(num_points, ndofs1)`.
        std::span<const T> trial_phi() const { return phi1; }

        /// Trial basis reference gradients, `(num_points, ndofs1, tdim)`.
        std::span<const T> trial_dphi_ref() const { return dphi1; }

        /// Basis values of coefficient `c`, `(num_points, space_dim_c)`.
        std::span<const T> coeff_phi(int c) const { return cphi[c]; }

        /// Offset of coefficient `c` in the packed coefficient array.
        int coeff_offset(int c) const { return offsets[c]; }

        /// Number of coefficients.
        int num_coeffs() const { return static_cast<int>(cphi.size()); }

        /// Number of quadrature points.
        int num_points() const { return static_cast<int>(nq); }

        /// Topological dimension.
        int tdim() const { return _tdim; }

        /// Number of test dofs per cell.
        int num_dofs0() const { return _ndofs0; }

        /// Number of trial dofs per cell.
        int num_dofs1() const { return _ndofs1; }

        /// Coordinate element reference basis values,
        /// `(num_points, ngeom_dofs)`.
        std::span<const T> coord_phi() const { return coord_phi_v; }

        /// Coordinate element reference basis gradients,
        /// `(num_points, ngeom_dofs, tdim)`.
        std::span<const T> coord_dphi_ref() const { return coord_dphi; }

        /// Number of coordinate dofs per cell.
        int num_geom_dofs() const { return _ngeom; }

        /// Geometric dimension (always 3 for the stored geometry).
        static constexpr int gdim() { return 3; }

    private:
        // Quadrature points and weights
        std::vector<T> Xq;
        std::vector<T> wq;
        std::size_t nq;

        // Test/trial basis at quadrature points
        std::vector<T> phi0, dphi0;
        std::vector<T> phi1, dphi1;

        // Coefficient basis values at quadrature points
        std::vector<std::vector<T>> cphi;

        // Cumulative space dimensions of the coefficients
        std::vector<int> offsets;

        // Coordinate element basis at quadrature points
        std::vector<T> coord_phi_v;
        std::vector<T> coord_dphi;

        int _tdim;
        int _ndofs0;
        int _ndofs1;
        int _ngeom;
    };

    /// Wrap a friendly weak form into a dolfinx-style cell `kernel_t`.
    ///
    /// The returned kernel computes, per cell, the Jacobian and its
    /// determinant from the cell geometry, maps the reference basis to
    /// physical values, evaluates the packed coefficients at the
    /// quadrature points and invokes `weak_fn` to accumulate the element
    /// tensor.
    ///
    /// @param[in] pre Precomputed reference data.
    /// @param[in] weak_fn The weak form.
    /// @return A cell kernel.
    template <std::floating_point T>
    kernel_t<T> make_cell_kernel(const PrecomputeData<T>& pre,
        cell_kernel_weak_fn_t<T> weak_fn);

} // namespace hellofem::fem
