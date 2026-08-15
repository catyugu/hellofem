// hellofem::fem — precomputed facet quadrature data and facet kernels
// SPDX-License-Identifier: MIT

#pragma once

#include "CoordinateElement.h"
#include "FiniteElement.h"
#include "kernel.h"
#include "mesh/cell_types.h"

#include <array>
#include <concepts>
#include <cstdint>
#include <span>
#include <vector>

namespace hellofem::fem {

    /// Per-facet quadrature data handed to a facet weak form.
    ///
    /// For a single-sided (exterior) facet every array holds `num_points`
    /// rows. For a double-sided (interior) facet the basis and coefficient
    /// arrays hold `2 * num_points` rows: rows `[0, num_points)` are the
    /// `+` cell side and `[num_points, 2*num_points)` the `-` side. The
    /// Jacobian determinant (facet measure) is shared and the normal
    /// points outward from the `+` cell, so `detJ` and `n` always have
    /// `num_points` entries.
    template <std::floating_point T>
    struct FacetKernelData {
        /// Test basis values, `(num_rows, ndofs0)` where `num_rows` is
        /// `num_points` (single-sided) or `2*num_points` (double-sided).
        std::span<const T> phi0;

        /// Test basis physical gradients, `(num_rows, ndofs0, tdim)`.
        std::span<const T> dphi0;

        /// Trial basis values.
        std::span<const T> phi1;

        /// Trial basis physical gradients.
        std::span<const T> dphi1;

        /// Quadrature weights (reference facet), `(num_points)`.
        std::span<const T> w;

        /// Facet measure at each quadrature point, `(num_points)`.
        std::span<const T> detJ;

        /// Scaled outward normal `normal * detJ`, `(num_points, gdim)`.
        /// Points outward from the `+` cell for double-sided facets.
        std::span<const T> n;

        /// Physical coordinates of the quadrature points,
        /// `(num_rows, gdim)`.
        std::span<const T> X;

        /// Coefficient values at the quadrature points,
        /// `(num_coeffs, num_rows)`, row `c` at `c * num_rows`.
        std::span<const T> coeffs;

        /// Physical gradients of the coefficients, `(num_coeffs,
        /// num_rows, gdim)`.
        std::span<const T> dcoeffs;

        /// Concatenated constant values in form order.
        const T* constants;

        /// Number of quadrature points per side.
        int num_points;

        /// Number of test dofs per cell.
        int num_dofs0;

        /// Number of trial dofs per cell.
        int num_dofs1;

        /// Topological dimension of the cell.
        int tdim;

        /// 0 for single-sided (exterior) facets, 1 for double-sided
        /// (interior) facets.
        int restricted;
    };

    /// A facet weak form: accumulate the facet element tensor `Ae`.
    ///
    /// For a single-sided facet `Ae` has `num_dofs0 * num_dofs1` entries
    /// (bilinear) or `num_dofs0` (linear). For a double-sided facet `Ae`
    /// has `2*num_dofs0 * 2*num_dofs1` (bilinear) or `2*num_dofs0`
    /// (linear), the blocks ordered side0-side0, side0-side1, side1-side0,
    /// side1-side1 row-major.
    template <std::floating_point T>
    using facet_kernel_weak_fn_t = void (*)(T* Ae,
        const FacetKernelData<T>& data);

    /// Reference-cell data for assembling facet integrals.
    ///
    /// Holds the quadrature rule on the reference facet and, for every
    /// local facet of the cell, the reference-cell coordinates of the
    /// quadrature points and the reference basis values of the test,
    /// trial and coefficient elements (and of the coordinate element) at
    /// those points, together with the constant per-facet reference
    /// jacobians and outward normals.
    template <std::floating_point T>
    class FacetPrecomputeData {
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
        FacetPrecomputeData(mesh::CellType cell_type,
            const FiniteElement<T>& test_element,
            const FiniteElement<T>& trial_element,
            const std::vector<const FiniteElement<T>*>& coeff_elements,
            const CoordinateElement<T>& coord_element, int quadrature_degree);

        /// Reference facet quadrature weights, `(num_points)`.
        std::span<const T> weights() const { return wq; }

        /// Test basis values of facet `f`, `(num_points, ndofs0)`.
        std::span<const T> test_phi(int f) const { return phi0[f]; }

        /// Test basis reference gradients of facet `f`.
        std::span<const T> test_dphi_ref(int f) const { return dphi0[f]; }

        /// Trial basis values of facet `f`.
        std::span<const T> trial_phi(int f) const { return phi1[f]; }

        /// Trial basis reference gradients of facet `f`.
        std::span<const T> trial_dphi_ref(int f) const { return dphi1[f]; }

        /// Basis values of coefficient `c` on facet `f`.
        std::span<const T> coeff_phi(int f, int c) const { return cphi[f][c]; }

        /// Reference gradients of coefficient `c` on facet `f`.
        std::span<const T> coeff_dphi(int f, int c) const { return cdphi[f][c]; }

        /// Offset of coefficient `c` in the packed coefficient array.
        int coeff_offset(int c) const { return offsets[c]; }

        /// Number of coefficients.
        int num_coeffs() const { return static_cast<int>(offsets.size()) - 1; }

        /// Number of quadrature points (per side).
        int num_points() const { return static_cast<int>(nq); }

        /// Topological dimension of the cell.
        int tdim() const { return _tdim; }

        /// Number of test dofs per cell.
        int num_dofs0() const { return _ndofs0; }

        /// Number of trial dofs per cell.
        int num_dofs1() const { return _ndofs1; }

        /// Number of coordinate dofs per cell.
        int num_geom_dofs() const { return _ngeom; }

        /// Geometric dimension.
        static constexpr int gdim() { return 3; }

        /// Number of facets of the cell.
        int num_facets() const { return static_cast<int>(J_ref.size()); }

        /// Reference jacobian of facet `f`, `(tdim, tdim-1)` row-major.
        std::span<const T> facet_jacobian(int f) const { return J_ref[f]; }

        /// Reference cell coordinate of the first vertex of facet `f`.
        std::array<T, 3> facet_origin(int f) const { return X0[f]; }

        /// Outward unit normal of facet `f` in reference coordinates,
        /// `(tdim)`.
        std::span<const T> facet_normal(int f) const { return n_ref[f]; }

        /// Coordinate element reference basis values of facet `f`,
        /// `(num_points, ngeom_dofs)`.
        std::span<const T> coord_phi(int f) const { return coord_phi_v[f]; }

        /// Coordinate element reference gradients of facet `f`,
        /// `(num_points, ngeom_dofs, tdim)`.
        std::span<const T> coord_dphi_ref(int f) const { return coord_dphi[f]; }

    private:
        // Quadrature weights on the reference facet
        std::vector<T> wq;
        std::size_t nq;

        // Test/trial basis at each facet's points: [facet][...]
        std::vector<std::vector<T>> phi0, dphi0;
        std::vector<std::vector<T>> phi1, dphi1;

        // Coefficient basis values and gradients: [facet][coeff][...]
        std::vector<std::vector<std::vector<T>>> cphi;
        std::vector<std::vector<std::vector<T>>> cdphi;

        // Cumulative space dimensions of the coefficients
        std::vector<int> offsets;

        // Coordinate element basis at each facet's points: [facet][...]
        std::vector<std::vector<T>> coord_phi_v;
        std::vector<std::vector<T>> coord_dphi;

        // Per-facet reference jacobians (tdim * (tdim-1)), outward unit
        // normals (tdim) and origin vertex coordinate
        std::vector<std::vector<T>> J_ref;
        std::vector<std::vector<T>> n_ref;
        std::vector<std::array<T, 3>> X0;

        int _tdim;
        int _ndofs0;
        int _ndofs1;
        int _ngeom;
    };

    /// Wrap a friendly single-sided facet weak form into a `kernel_t`.
    ///
    /// The returned kernel computes, per exterior-facet entity (cell,
    /// local_facet), the facet measure, scaled outward normal and physical
    /// basis values from the cell geometry, then invokes `weak_fn` to
    /// accumulate the facet element tensor.
    ///
    /// @param[in] pre Precomputed reference data.
    /// @param[in] weak_fn The facet weak form.
    /// @return A facet kernel.
    template <std::floating_point T>
    kernel_t<T> make_facet_kernel(const FacetPrecomputeData<T>& pre,
        facet_kernel_weak_fn_t<T> weak_fn);

    /// Wrap a friendly double-sided facet weak form into a `kernel_t`.
    ///
    /// The returned kernel computes the geometry of both cells sharing an
    /// interior facet, fills the double-sided data (`2*num_points` rows,
    /// shared normal/measure pointing outward from the `+` cell) and
    /// invokes `weak_fn` to accumulate the facet element tensor.
    ///
    /// @param[in] pre Precomputed reference data.
    /// @param[in] weak_fn The facet weak form.
    /// @return A facet kernel.
    template <std::floating_point T>
    kernel_t<T> make_interior_facet_kernel(const FacetPrecomputeData<T>& pre,
        facet_kernel_weak_fn_t<T> weak_fn);

} // namespace hellofem::fem
