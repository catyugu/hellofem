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

        /// Physical gradients of the coefficients at the quadrature
        /// points, `(num_coeffs, num_points, gdim)`. Only filled by
        /// `make_expression_kernel`; empty for `make_cell_kernel`.
        std::span<const T> dcoeffs;

        /// Concatenated constant values in form order. The weak form
        /// knows its own constant count.
        const T* constants;

        /// Physical coordinates of the quadrature points,
        /// `(num_points, gdim)`, row `p` at `p * gdim`. Only filled by
        /// `make_expression_kernel`; empty for `make_cell_kernel`.
        std::span<const T> X;

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

        /// Reference gradients of coefficient `c`,
        /// `(num_points, space_dim_c, tdim)`.
        std::span<const T> coeff_dphi(int c) const { return cdphi[c]; }

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

        // Coefficient reference gradients at quadrature points
        std::vector<std::vector<T>> cdphi;

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

    /// Wrap an expression writer into a dolfinx-style `kernel_t`.
    ///
    /// Unlike `make_cell_kernel`, the returned kernel fills the output
    /// buffer `Ae` with the value of an expression at each quadrature
    /// point (one value per point, `num_points * value_size` entries)
    /// rather than accumulating an element tensor. The writer receives
    /// the friendly data with the physical point coordinates filled in
    /// `data.X`, physical coefficient values in `data.coeffs`, their
    /// gradients in `data.dcoeffs`, and physical test/trial gradients
    /// in `data.dphi0`/`data.dphi1`.
    ///
    /// @tparam Fn Writer type; any callable `void(T*, const
    /// CellKernelData<T>&)`.
    /// @param[in] pre Precomputed reference data.
    /// @param[in] writer The expression writer.
    /// @return An expression kernel.
    template <std::floating_point T, typename Fn>
    kernel_t<T> make_expression_kernel(const PrecomputeData<T>& pre, Fn writer)
    {
        const int nq = pre.num_points();
        const int tdim = pre.tdim();
        const int ndofs0 = pre.num_dofs0();
        const int ndofs1 = pre.num_dofs1();
        const int ncoeffs = pre.num_coeffs();
        const int ngeom = pre.num_geom_dofs();

        // Per-cell scratch buffers, allocated once.
        constexpr int gdim = 3;
        std::vector<T> cdofs(static_cast<std::size_t>(ngeom) * gdim);
        std::vector<T> J(static_cast<std::size_t>(gdim) * tdim);
        std::vector<T> K(static_cast<std::size_t>(tdim) * gdim);
        std::vector<T> detJ(nq);
        std::vector<T> Jwork(2 * gdim * tdim);
        std::vector<T> dphi0_phys(static_cast<std::size_t>(nq) * ndofs0 * tdim);
        std::vector<T> dphi1_phys(static_cast<std::size_t>(nq) * ndofs1 * tdim);
        std::vector<T> coeffs_phys(static_cast<std::size_t>(ncoeffs) * nq);
        std::vector<T> dcoeffs_phys(
            static_cast<std::size_t>(ncoeffs) * nq * gdim);
        std::vector<T> X_phys(static_cast<std::size_t>(nq) * gdim);

        return [&pre, writer = std::move(writer), nq, tdim, ndofs0, ndofs1,
                   ncoeffs, ngeom, cdofs, J, K, detJ, Jwork, dphi0_phys,
                   dphi1_phys, coeffs_phys, dcoeffs_phys, X_phys](
                   T* Ae, const T* coeffs, const T* constants,
                   const double* cds, const int*, const std::uint8_t*, void*) mutable {
            const std::size_t nq_ = static_cast<std::size_t>(nq);
            const std::size_t nd_ = static_cast<std::size_t>(ngeom);
            constexpr int gdim = 3;

            // Gather the cell geometry nodes: (ngeom, gdim).
            for (int j = 0; j < ngeom; ++j)
                for (int k = 0; k < gdim; ++k)
                    cdofs[j * gdim + k] = cds[j * gdim + k];

            for (int p = 0; p < nq; ++p) {
                // Jacobian and its inverse/determinant at the point.
                std::fill(J.begin(), J.end(), 0);
                for (int i = 0; i < gdim; ++i)
                    for (int k = 0; k < tdim; ++k) {
                        T acc = 0;
                        for (int j = 0; j < ngeom; ++j)
                            acc += pre.coord_dphi_ref()[((p * nd_ + j)) * tdim + k]
                                * cdofs[j * gdim + i];
                        J[i * tdim + k] = acc;
                    }

                md::mdspan<T, md::dextents<std::size_t, 2>> Jm(J.data(), gdim, tdim);
                md::mdspan<T, md::dextents<std::size_t, 2>> Km(K.data(), tdim, gdim);
                if (tdim == gdim)
                    math::inv(Jm, Km);
                else
                    math::pinv(Jm, Km);

                if (tdim == gdim)
                    detJ[p] = std::abs(math::det(Jm));
                else
                    detJ[p] = std::abs(
                        CoordinateElement<T>::compute_jacobian_determinant(
                            Jm, std::span(Jwork.data(), Jwork.size())));

                // Physical coordinates: X[p] = sum_j coord_phi[p][j] * cdofs[j].
                for (int i = 0; i < gdim; ++i) {
                    T acc = 0;
                    for (int j = 0; j < ngeom; ++j)
                        acc += pre.coord_phi()[p * ngeom + j] * cdofs[j * gdim + i];
                    X_phys[p * gdim + i] = acc;
                }

                // Physical gradients of test/trial basis: dphi * K.
                for (int i = 0; i < ndofs0; ++i)
                    for (int c = 0; c < tdim; ++c) {
                        T acc = 0;
                        for (int j = 0; j < tdim; ++j)
                            acc += pre.test_dphi_ref()[((p * ndofs0 + i)) * tdim + j]
                                * K[j * gdim + c];
                        dphi0_phys[(p * ndofs0 + i) * tdim + c] = acc;
                    }
                for (int i = 0; i < ndofs1; ++i)
                    for (int c = 0; c < tdim; ++c) {
                        T acc = 0;
                        for (int j = 0; j < tdim; ++j)
                            acc += pre.trial_dphi_ref()[((p * ndofs1 + i)) * tdim + j]
                                * K[j * gdim + c];
                        dphi1_phys[(p * ndofs1 + i) * tdim + c] = acc;
                    }

                // Coefficient values at the point.
                for (int c = 0; c < ncoeffs; ++c) {
                    const int offset = pre.coeff_offset(c);
                    const std::size_t csize = pre.coeff_phi(c).size() / nq_;
                    T acc = 0;
                    for (std::size_t i = 0; i < csize; ++i)
                        acc += pre.coeff_phi(c)[p * csize + i] * coeffs[offset + i];
                    coeffs_phys[c * nq_ + p] = acc;
                }

                // Physical coefficient gradients: dcoeffs = dphi_ref * K.
                for (int c = 0; c < ncoeffs; ++c) {
                    const int offset = pre.coeff_offset(c);
                    const auto cdphi = pre.coeff_dphi(c);
                    const std::size_t csize_ref = cdphi.size() / (nq_ * tdim);
                    for (int q = 0; q < gdim; ++q) {
                        T gacc = 0;
                        for (int j = 0; j < tdim; ++j)
                            for (std::size_t i = 0; i < csize_ref; ++i)
                                gacc += cdphi[(p * csize_ref + i) * tdim + j]
                                    * K[j * gdim + q] * coeffs[offset + i];
                        dcoeffs_phys[(c * nq_ + p) * gdim + q] = gacc;
                    }
                }
            }

            // Build the friendly data view and invoke the writer.
            CellKernelData<T> data;
            data.phi0 = pre.test_phi();
            data.dphi0 = dphi0_phys;
            data.phi1 = pre.trial_phi();
            data.dphi1 = dphi1_phys;
            data.w = pre.weights();
            data.detJ = detJ;
            data.coeffs = coeffs_phys;
            data.dcoeffs = dcoeffs_phys;
            data.constants = constants;
            data.X = X_phys;
            data.num_points = nq;
            data.num_dofs0 = ndofs0;
            data.num_dofs1 = ndofs1;
            data.tdim = tdim;
            writer(Ae, data);
        };
    }

} // namespace hellofem::fem
