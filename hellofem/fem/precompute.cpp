// hellofem::fem — precomputed quadrature data and cell-kernel factory
// SPDX-License-Identifier: MIT

#include "precompute.h"

#include "basis/quadrature.h"
#include "common/math.h"

#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

namespace hellofem::fem {

    namespace {

        /// Evaluate the reference basis and first derivatives of a scalar
        /// element at a set of points. `phi` is filled with values
        /// `(num_points, num_dofs)`, `dphi` with gradients
        /// `(num_points, num_dofs, tdim)`.
        ///
        /// For a blocked element (block size `bs`) the scalar sub-element
        /// basis is tabulated once and replicated across the block with
        /// node-major dof ordering (`dof = scalar_dof * bs + component`).
        template <std::floating_point T>
        void _tabulate_scalar_basis(const FiniteElement<T>& element,
            const std::vector<T>& Xq, std::array<std::size_t, 2> xshape,
            std::vector<T>& phi, std::vector<T>& dphi)
        {
            const int num_points = static_cast<int>(xshape[0]);
            const int tdim = static_cast<int>(xshape[1]);
            const int ndofs = element.space_dimension();
            const int bs = element.block_size();
            const FiniteElement<T>& base
                = bs > 1 ? *element.sub_elements()[0] : element;
            const int ndofs_scalar = base.space_dimension();

            // Allocating tabulate returns (data, {nderivs, nq, ndofs, vs}).
            auto [basis, shape] = base.tabulate(Xq, xshape, 1);
            (void)shape;

            const std::size_t np = static_cast<std::size_t>(num_points);
            const std::size_t ns = static_cast<std::size_t>(ndofs_scalar);
            const std::size_t nd = static_cast<std::size_t>(ndofs);
            const std::size_t nbs = static_cast<std::size_t>(bs);
            phi.assign(np * nd, 0);
            dphi.assign(np * nd * static_cast<std::size_t>(tdim), 0);
            for (std::size_t p = 0; p < np; ++p) {
                for (std::size_t s = 0; s < ns; ++s) {
                    for (std::size_t c = 0; c < nbs; ++c) {
                        const std::size_t d = s * nbs + c;
                        phi[p * nd + d] = basis[p * ns + s];
                        for (int j = 0; j < tdim; ++j)
                            dphi[(p * nd + d) * tdim + j]
                                = basis[(static_cast<std::size_t>(j + 1) * np + p) * ns + s];
                    }
                }
            }
        }

        /// Coordinate element reference basis and derivatives at the
        /// quadrature points. The coordinate element is scalar (one
        /// component), so its tabulate fills `(nderivs, nq, ngeom, 1)`.
        template <std::floating_point T>
        void _tabulate_coord_basis(const CoordinateElement<T>& coord_element,
            const std::vector<T>& Xq, std::array<std::size_t, 2> xshape,
            std::vector<T>& phi, std::vector<T>& dphi)
        {
            const int num_points = static_cast<int>(xshape[0]);
            const int tdim = static_cast<int>(xshape[1]);
            const int ngeom = coord_element.dim();

            auto shape = coord_element.tabulate_shape(1, num_points);
            std::vector<T> basis(shape[0] * shape[1] * shape[2] * shape[3]);
            coord_element.tabulate(1, Xq, xshape, std::span(basis));

            const std::size_t np = static_cast<std::size_t>(num_points);
            const std::size_t ng = static_cast<std::size_t>(ngeom);
            phi.assign(np * ng, 0);
            dphi.assign(np * ng * static_cast<std::size_t>(tdim), 0);
            for (std::size_t p = 0; p < np; ++p) {
                for (std::size_t i = 0; i < ng; ++i) {
                    phi[p * ng + i] = basis[p * ng + i];
                    for (int j = 0; j < tdim; ++j)
                        dphi[(p * ng + i) * tdim + j]
                            = basis[(static_cast<std::size_t>(j + 1) * np + p) * ng + i];
                }
            }
        }

    } // namespace

    template <std::floating_point T>
    PrecomputeData<T>::PrecomputeData(mesh::CellType cell_type,
        const FiniteElement<T>& test_element,
        const FiniteElement<T>& trial_element,
        const std::vector<const FiniteElement<T>*>& coeff_elements,
        const CoordinateElement<T>& coord_element, int quadrature_degree)
    {
        _tdim = mesh::cell_dim(cell_type);
        _ndofs0 = test_element.space_dimension();
        _ndofs1 = trial_element.space_dimension();
        _ngeom = coord_element.dim();

        // Quadrature rule on the reference cell.
        auto [Xq_data, wq_data] = basis::quadrature::make_quadrature<T>(
            basis::quadrature::type::Default,
            mesh::cell_type_to_basix_type(cell_type),
            basis::polyset::type::standard, quadrature_degree);
        Xq = std::move(Xq_data);
        wq = std::move(wq_data);
        nq = wq.size();
        const std::array<std::size_t, 2> xshape {
            nq, static_cast<std::size_t>(_tdim)};

        // Test/trial/coefficient basis at the quadrature points.
        _tabulate_scalar_basis(test_element, Xq, xshape, phi0, dphi0);
        _tabulate_scalar_basis(trial_element, Xq, xshape, phi1, dphi1);
        for (const auto* ce : coeff_elements) {
            std::vector<T> phi, dphi;
            _tabulate_scalar_basis(*ce, Xq, xshape, phi, dphi);
            cphi.push_back(std::move(phi));
            cdphi.push_back(std::move(dphi));
        }
        offsets.push_back(0);
        for (const auto* ce : coeff_elements)
            offsets.push_back(offsets.back() + ce->space_dimension());

        // Coordinate element basis at the quadrature points.
        _tabulate_coord_basis(coord_element, Xq, xshape, coord_phi_v,
            coord_dphi);
    }

    template <std::floating_point T>
    kernel_t<T> make_cell_kernel(const PrecomputeData<T>& pre,
        cell_kernel_weak_fn_t<T> weak_fn)
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

        return [&pre, weak_fn, nq, tdim, ndofs0, ndofs1, ncoeffs, ngeom,
                   cdofs, J, K, detJ, Jwork, dphi0_phys, dphi1_phys,
                   coeffs_phys, dcoeffs_phys](
                   T* Ae, const T* coeffs, const T* constants,
                   const double* cds, const int*, const std::uint8_t*, void*) mutable {
            const std::size_t nq_ = static_cast<std::size_t>(nq);
            const std::size_t nd_ = static_cast<std::size_t>(ngeom);
            constexpr int gdim = 3;

            // Gather the cell geometry nodes: (ngeom, gdim).
            for (int j = 0; j < ngeom; ++j)
                for (int k = 0; k < gdim; ++k)
                    cdofs[j * gdim + k] = cds[j * gdim + k];

            // Compute the Jacobian at each quadrature point and map the
            // reference gradients to physical ones.
            for (int p = 0; p < nq; ++p) {
                // J[i][k] = sum_j coord_dphi[p][j][k] * cdofs[j][i].
                std::fill(J.begin(), J.end(), 0);
                for (int i = 0; i < gdim; ++i)
                    for (int k = 0; k < tdim; ++k) {
                        T acc = 0;
                        for (int j = 0; j < ngeom; ++j)
                            acc += pre.coord_dphi_ref()[((p * nd_ + j)) * tdim + k]
                                * cdofs[j * gdim + i];
                        J[i * tdim + k] = acc;
                    }

                // Inverse Jacobian (pseudo-inverse for tdim < gdim) and
                // area factor.
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

                // Coefficient values at point p: sum_i cphi_c[p][i] * coeffs[offset_c + i].
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

            // Build the friendly data view and invoke the weak form.
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
            data.num_points = nq;
            data.num_dofs0 = ndofs0;
            data.num_dofs1 = ndofs1;
            data.tdim = tdim;
            weak_fn(Ae, data);
        };
    }

    template class PrecomputeData<double>;
    template class PrecomputeData<float>;

    template kernel_t<double> make_cell_kernel(const PrecomputeData<double>&,
        cell_kernel_weak_fn_t<double>);
    template kernel_t<float> make_cell_kernel(const PrecomputeData<float>&,
        cell_kernel_weak_fn_t<float>);

} // namespace hellofem::fem
