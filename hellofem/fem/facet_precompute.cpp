// hellofem::fem — precomputed facet quadrature data and facet kernels
// SPDX-License-Identifier: MIT

#include "facet_precompute.h"

#include "basis/cell.h"
#include "basis/quadrature.h"
#include "common/math.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <span>
#include <vector>

namespace hellofem::fem {

    namespace {

        /// Reference-cell coordinates of the quadrature points of a facet
        /// `f` of a cell: the reference facet points (tdim-1 coordinates)
        /// mapped into the reference cell via the facet's affine map
        /// `X = X0 + J_ref * X_facet`.
        template <std::floating_point T>
        std::vector<T> _facet_cell_points(mesh::CellType cell_type, int f,
            const std::vector<T>& Xq, std::size_t nq)
        {
            const int tdim = mesh::cell_dim(cell_type);
            const auto [X0, X0shape] = basis::cell::sub_entity_geometry<T>(
                mesh::cell_type_to_basix_type(cell_type), tdim - 1, f);
            const auto [J_ref, Jshape]
                = basis::cell::facet_jacobians<T>(
                    mesh::cell_type_to_basix_type(cell_type));
            // J_ref: (nfacets, tdim, tdim-1); X0: (nverts, tdim).
            std::vector<T> Xc(nq * tdim, 0);
            for (std::size_t p = 0; p < nq; ++p)
                for (int i = 0; i < tdim; ++i) {
                    T acc = X0[i]; // first vertex of the facet
                    for (int j = 0; j < tdim - 1; ++j)
                        acc += J_ref[static_cast<std::size_t>(f) * tdim * (tdim - 1)
                                   + i * (tdim - 1) + j]
                            * Xq[p * (tdim - 1) + j];
                    Xc[p * tdim + i] = acc;
                }
            return Xc;
        }

        /// Reference basis values and derivatives of a scalar element at a
        /// set of reference points. `phi` is `(num_points, num_dofs)`,
        /// `dphi` `(num_points, num_dofs, tdim)`.
        template <std::floating_point T>
        void _tabulate_scalar_basis(const FiniteElement<T>& element,
            const std::vector<T>& Xq, std::array<std::size_t, 2> xshape,
            std::vector<T>& phi, std::vector<T>& dphi)
        {
            const int num_points = static_cast<int>(xshape[0]);
            const int tdim = static_cast<int>(xshape[1]);
            const int ndofs = element.space_dimension();

            auto [basis, shape] = element.tabulate(Xq, xshape, 1);
            (void)shape;

            const std::size_t np = static_cast<std::size_t>(num_points);
            const std::size_t nd = static_cast<std::size_t>(ndofs);
            phi.assign(np * nd, 0);
            dphi.assign(np * nd * static_cast<std::size_t>(tdim), 0);
            for (std::size_t p = 0; p < np; ++p) {
                for (std::size_t i = 0; i < nd; ++i) {
                    phi[p * nd + i] = basis[p * nd + i];
                    for (int j = 0; j < tdim; ++j)
                        dphi[(p * nd + i) * tdim + j]
                            = basis[(static_cast<std::size_t>(j + 1) * np + p) * nd + i];
                }
            }
        }

        /// Coordinate element reference basis and derivatives at a set of
        /// points.
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
    FacetPrecomputeData<T>::FacetPrecomputeData(mesh::CellType cell_type,
        const FiniteElement<T>& test_element,
        const FiniteElement<T>& trial_element,
        const std::vector<const FiniteElement<T>*>& coeff_elements,
        const CoordinateElement<T>& coord_element, int quadrature_degree)
    {
        _tdim = mesh::cell_dim(cell_type);
        _ndofs0 = test_element.space_dimension();
        _ndofs1 = trial_element.space_dimension();
        _ngeom = coord_element.dim();

        const basis::cell::type bcell = mesh::cell_type_to_basix_type(cell_type);

        // Quadrature rule on the reference facet.
        const mesh::CellType facet_type
            = mesh::cell_entity_type(cell_type, _tdim - 1, 0);
        auto [Xq_data, wq_data] = basis::quadrature::make_quadrature<T>(
            basis::quadrature::type::Default,
            mesh::cell_type_to_basix_type(facet_type),
            basis::polyset::type::standard, quadrature_degree);
        wq = std::move(wq_data);
        nq = wq.size();

        // Per-facet reference jacobians, normals and origin vertex.
        const auto [J_data, Jshape] = basis::cell::facet_jacobians<T>(bcell);
        // Jshape: (nfacets, tdim, tdim-1).
        const std::size_t nfacets = Jshape[0];
        const std::size_t J_size = static_cast<std::size_t>(_tdim) * (_tdim - 1);
        J_ref.resize(nfacets);
        for (std::size_t f = 0; f < nfacets; ++f)
            J_ref[f].assign(
                J_data.begin() + static_cast<std::int64_t>(f * J_size),
                J_data.begin() + static_cast<std::int64_t>((f + 1) * J_size));

        const auto [n_data, nshape] = basis::cell::facet_outward_normals<T>(bcell);
        // nshape: (nfacets, tdim).
        n_ref.resize(nfacets);
        for (std::size_t f = 0; f < nfacets; ++f)
            n_ref[f].assign(
                n_data.begin() + static_cast<std::int64_t>(f * _tdim),
                n_data.begin() + static_cast<std::int64_t>((f + 1) * _tdim));

        X0.resize(nfacets);
        for (std::size_t f = 0; f < nfacets; ++f) {
            const auto [X0_data, X0shape] = basis::cell::sub_entity_geometry<T>(
                bcell, _tdim - 1, static_cast<int>(f));
            std::array<T, 3> origin {};
            for (std::size_t i = 0; i < static_cast<std::size_t>(_tdim); ++i)
                origin[i] = X0_data[i];
            X0[f] = origin;
        }

        // Basis data per facet.
        phi0.resize(nfacets);
        dphi0.resize(nfacets);
        phi1.resize(nfacets);
        dphi1.resize(nfacets);
        coord_phi_v.resize(nfacets);
        coord_dphi.resize(nfacets);
        cphi.resize(nfacets);
        cdphi.resize(nfacets);

        for (std::size_t f = 0; f < nfacets; ++f) {
            std::vector<T> Xc = _facet_cell_points<T>(cell_type, static_cast<int>(f), Xq_data, nq);
            const std::array<std::size_t, 2> xshape_cell {
                nq, static_cast<std::size_t>(_tdim)};

            _tabulate_scalar_basis(test_element, Xc, xshape_cell, phi0[f], dphi0[f]);
            _tabulate_scalar_basis(trial_element, Xc, xshape_cell, phi1[f], dphi1[f]);
            for (const auto* ce : coeff_elements) {
                std::vector<T> phi, dphi;
                _tabulate_scalar_basis(*ce, Xc, xshape_cell, phi, dphi);
                cphi[f].push_back(std::move(phi));
                cdphi[f].push_back(std::move(dphi));
            }
            _tabulate_coord_basis(coord_element, Xc, xshape_cell, coord_phi_v[f], coord_dphi[f]);
        }

        offsets.push_back(0);
        for (const auto* ce : coeff_elements)
            offsets.push_back(offsets.back() + ce->space_dimension());
    }

    // ------------------------------------------------------------------ //

    namespace {

        /// Compute the facet geometry for one cell: the facet jacobian
        /// (physical), its measure and the scaled outward normal.
        template <std::floating_point T>
        void _facet_geometry(const FacetPrecomputeData<T>& pre, int local_facet,
            const std::vector<T>& cdofs, int nq, std::vector<T>& detJ,
            std::vector<T>& n_phys, std::vector<T>& X_phys)
        {
            const int tdim = pre.tdim();
            const int ngeom = pre.num_geom_dofs();
            constexpr int gdim = 3;

            const int nf = tdim - 1; // facet topological dimension
            const auto J_ref = pre.facet_jacobian(local_facet); // tdim*nf
            const auto n_ref = pre.facet_normal(local_facet);   // tdim

            // Reference coordinates of the facet points were used to build
            // the basis; the physical map is evaluated at those points.
            const auto coord_dphi = pre.coord_dphi_ref(local_facet);
            const auto coord_phi = pre.coord_phi(local_facet);

            std::vector<T> J(gdim * tdim), K(tdim * gdim), Jw(gdim * nf);
            std::vector<T> Jf(gdim * nf), JfTJf(nf * nf), JfTJf_work(nf * nf);
            for (int p = 0; p < nq; ++p) {
                // Cell jacobian J at the facet point.
                std::fill(J.begin(), J.end(), 0);
                for (int i = 0; i < gdim; ++i)
                    for (int k = 0; k < tdim; ++k) {
                        T acc = 0;
                        for (int j = 0; j < ngeom; ++j)
                            acc += coord_dphi[static_cast<std::size_t>((p * ngeom + j)) * tdim + k]
                                * cdofs[j * gdim + i];
                        J[i * tdim + k] = acc;
                    }
                md::mdspan<T, md::dextents<std::size_t, 2>> Jm(J.data(), gdim, tdim);
                md::mdspan<T, md::dextents<std::size_t, 2>> Km(K.data(), tdim, gdim);
                if (tdim == gdim)
                    math::inv(Jm, Km);
                else
                    math::pinv(Jm, Km);

                // Physical facet jacobian Jf = J * J_ref (gdim x nf).
                for (int i = 0; i < gdim; ++i)
                    for (int j = 0; j < nf; ++j) {
                        T acc = 0;
                        for (int k = 0; k < tdim; ++k)
                            acc += J[i * tdim + k] * J_ref[k * nf + j];
                        Jf[i * nf + j] = acc;
                    }

                // Facet measure: sqrt(det(Jf^T Jf)).
                for (int i = 0; i < nf; ++i)
                    for (int j = 0; j < nf; ++j) {
                        T acc = 0;
                        for (int k = 0; k < gdim; ++k)
                            acc += Jf[k * nf + i] * Jf[k * nf + j];
                        JfTJf[i * nf + j] = acc;
                    }
                detJ[p] = std::sqrt(math::det(md::mdspan<const T, md::dextents<std::size_t, 2>>(
                    JfTJf.data(), nf, nf)));

                // Scaled outward normal: K^T n_ref * detJ.
                for (int i = 0; i < gdim; ++i) {
                    T acc = 0;
                    for (int k = 0; k < tdim; ++k)
                        acc += K[k * gdim + i] * n_ref[k];
                    n_phys[p * gdim + i] = acc * detJ[p];
                }

                // Physical coordinates of the point.
                for (int i = 0; i < gdim; ++i) {
                    T acc = 0;
                    for (int j = 0; j < ngeom; ++j)
                        acc += coord_phi[static_cast<std::size_t>(p * ngeom + j)]
                            * cdofs[j * gdim + i];
                    X_phys[p * gdim + i] = acc;
                }
            }
        }

        /// Map reference basis gradients to physical ones: dphi_phys =
        /// dphi_ref * K, per point.
        template <std::floating_point T>
        void _map_gradients(const std::span<const T> dphi_ref, int nq,
            int ndofs, int tdim, int gdim, const std::vector<T>& K,
            std::vector<T>& dphi_phys)
        {
            dphi_phys.assign(static_cast<std::size_t>(nq) * ndofs * gdim, 0);
            for (int p = 0; p < nq; ++p)
                for (int i = 0; i < ndofs; ++i)
                    for (int c = 0; c < gdim; ++c) {
                        T acc = 0;
                        for (int j = 0; j < tdim; ++j)
                            acc += dphi_ref[static_cast<std::size_t>((p * ndofs + i)) * tdim + j]
                                * K[j * gdim + c];
                        dphi_phys[static_cast<std::size_t>((p * ndofs + i)) * gdim + c] = acc;
                    }
        }

    } // namespace

    template <std::floating_point T>
    kernel_t<T> make_facet_kernel(const FacetPrecomputeData<T>& pre,
        facet_kernel_weak_fn_t<T> weak_fn)
    {
        const int nq = pre.num_points();
        const int tdim = pre.tdim();
        const int ndofs0 = pre.num_dofs0();
        const int ndofs1 = pre.num_dofs1();
        const int ncoeffs = pre.num_coeffs();
        const int ngeom = pre.num_geom_dofs();
        constexpr int gdim = 3;

        std::vector<T> cdofs(static_cast<std::size_t>(ngeom) * gdim);
        std::vector<T> detJ(nq);
        std::vector<T> n_phys(static_cast<std::size_t>(nq) * gdim);
        std::vector<T> X_phys(static_cast<std::size_t>(nq) * gdim);
        std::vector<T> dphi0_phys, dphi1_phys;
        std::vector<T> coeffs_phys(static_cast<std::size_t>(ncoeffs) * nq);
        std::vector<T> dcoeffs_phys(static_cast<std::size_t>(ncoeffs) * nq * gdim);

        return [&pre, weak_fn, nq, tdim, ndofs0, ndofs1, ncoeffs, ngeom,
                   cdofs, detJ, n_phys, X_phys, dphi0_phys, dphi1_phys,
                   coeffs_phys, dcoeffs_phys](
                   T* Ae, const T* coeffs, const T* constants,
                   const double* cds, const int* entity, const std::uint8_t*, void*) mutable {
            const int local_facet = entity ? *entity : 0;
            const std::size_t nq_ = static_cast<std::size_t>(nq);
            const std::size_t nd_ = static_cast<std::size_t>(ngeom);
            constexpr int gdim = 3;

            // Gather the cell geometry nodes: (ngeom, gdim).
            for (int j = 0; j < ngeom; ++j)
                for (int k = 0; k < gdim; ++k)
                    cdofs[j * gdim + k] = cds[j * gdim + k];

            // Compute facet measure, normal and physical coordinates.
            _facet_geometry(pre, local_facet, cdofs, nq, detJ, n_phys, X_phys);

            // Physical gradients of test/trial basis via the cell jacobian
            // inverse at the facet points (constant for affine geometry).
            std::vector<T> J(gdim * tdim), Km(tdim * gdim);
            const auto coord_dphi = pre.coord_dphi_ref(local_facet);
            std::fill(J.begin(), J.end(), 0);
            for (int i = 0; i < gdim; ++i)
                for (int k = 0; k < tdim; ++k) {
                    T acc = 0;
                    for (int j = 0; j < ngeom; ++j)
                        acc += coord_dphi[((0 * nd_ + j)) * tdim + k]
                            * cdofs[j * gdim + i];
                    J[i * tdim + k] = acc;
                }
            md::mdspan<T, md::dextents<std::size_t, 2>> Jm(J.data(), gdim, tdim);
            md::mdspan<T, md::dextents<std::size_t, 2>> Km_m(Km.data(), tdim, gdim);
            if (tdim == gdim)
                math::inv(Jm, Km_m);
            else
                math::pinv(Jm, Km_m);
            _map_gradients(pre.test_dphi_ref(local_facet), nq, ndofs0, tdim, gdim,
                Km, dphi0_phys);
            _map_gradients(pre.trial_dphi_ref(local_facet), nq, ndofs1, tdim, gdim,
                Km, dphi1_phys);

            // Coefficient values and physical gradients at the facet points.
            for (int c = 0; c < ncoeffs; ++c) {
                const int offset = pre.coeff_offset(c);
                const auto cphi_f = pre.coeff_phi(local_facet, c);
                const std::size_t csize = cphi_f.size() / nq_;
                for (int p = 0; p < nq; ++p) {
                    T acc = 0;
                    for (std::size_t i = 0; i < csize; ++i)
                        acc += cphi_f[static_cast<std::size_t>(p * csize + i)]
                            * coeffs[offset + i];
                    coeffs_phys[static_cast<std::size_t>(c) * nq_ + p] = acc;
                }
                const auto cdphi_f = pre.coeff_dphi(local_facet, c);
                const std::size_t csize_ref = cdphi_f.size() / (nq_ * tdim);
                for (int q = 0; q < gdim; ++q)
                    for (int p = 0; p < nq; ++p) {
                        T gacc = 0;
                        for (int j = 0; j < tdim; ++j)
                            for (std::size_t i = 0; i < csize_ref; ++i)
                                gacc += cdphi_f[static_cast<std::size_t>((p * csize_ref + i)) * tdim + j]
                                    * Km[j * gdim + q] * coeffs[offset + i];
                        dcoeffs_phys[static_cast<std::size_t>((c * nq_ + p)) * gdim + q] = gacc;
                    }
            }

            FacetKernelData<T> data;
            data.phi0 = pre.test_phi(local_facet);
            data.dphi0 = dphi0_phys;
            data.phi1 = pre.trial_phi(local_facet);
            data.dphi1 = dphi1_phys;
            data.w = pre.weights();
            data.detJ = detJ;
            data.n = n_phys;
            data.X = X_phys;
            data.coeffs = coeffs_phys;
            data.dcoeffs = dcoeffs_phys;
            data.constants = constants;
            data.num_points = nq;
            data.num_dofs0 = ndofs0;
            data.num_dofs1 = ndofs1;
            data.tdim = tdim;
            data.restricted = 0;
            weak_fn(Ae, data);
        };
    }

    template <std::floating_point T>
    kernel_t<T> make_interior_facet_kernel(const FacetPrecomputeData<T>& pre,
        facet_kernel_weak_fn_t<T> weak_fn)
    {
        const int nq = pre.num_points();
        const int tdim = pre.tdim();
        const int ndofs0 = pre.num_dofs0();
        const int ndofs1 = pre.num_dofs1();
        const int ncoeffs = pre.num_coeffs();
        const int ngeom = pre.num_geom_dofs();
        constexpr int gdim = 3;
        const int nrows = 2 * nq;

        std::vector<T> cdofs(static_cast<std::size_t>(2 * ngeom) * gdim);
        std::vector<T> detJ(nq);
        std::vector<T> n_phys(static_cast<std::size_t>(nq) * gdim);
        std::vector<T> X_phys(static_cast<std::size_t>(nrows) * gdim);
        std::vector<T> dphi0_phys, dphi1_phys;
        std::vector<T> coeffs_phys(static_cast<std::size_t>(ncoeffs) * nrows);
        std::vector<T> dcoeffs_phys(static_cast<std::size_t>(ncoeffs) * nrows * gdim);

        return [&pre, weak_fn, nq, nrows, tdim, ndofs0, ndofs1, ncoeffs,
                   ngeom, cdofs, detJ, n_phys, X_phys, dphi0_phys, dphi1_phys,
                   coeffs_phys, dcoeffs_phys](
                   T* Ae, const T* coeffs, const T* constants,
                   const double* cds, const int* entity, const std::uint8_t*, void*) mutable {
            const int local_facet0 = entity ? entity[0] : 0;
            const int local_facet1 = entity ? entity[1] : 0;
            const std::size_t nq_ = static_cast<std::size_t>(nq);
            const std::size_t nd_ = static_cast<std::size_t>(ngeom);
            constexpr int gdim = 3;

            // Gather both cells' geometry: cell0 at [0, ngeom), cell1 at
            // [ngeom, 2*ngeom).
            for (int side = 0; side < 2; ++side)
                for (int j = 0; j < ngeom; ++j)
                    for (int k = 0; k < gdim; ++k)
                        cdofs[(side * ngeom + j) * gdim + k]
                            = cds[(side * ngeom + j) * gdim + k];

            // Facet geometry from cell0 (normal points outward from cell0).
            std::vector<T> cdofs0(cdofs.begin(), cdofs.begin() + static_cast<std::int64_t>(ngeom * gdim));
            _facet_geometry(pre, local_facet0, cdofs0, nq, detJ, n_phys, X_phys);

            // Physical coordinates and gradients of both cells.
            std::vector<T> J(gdim * tdim), Km(tdim * gdim);
            for (int side = 0; side < 2; ++side) {
                const int local_facet = side == 0 ? local_facet0 : local_facet1;
                const auto coord_dphi = pre.coord_dphi_ref(local_facet);
                std::fill(J.begin(), J.end(), 0);
                for (int i = 0; i < gdim; ++i)
                    for (int k = 0; k < tdim; ++k) {
                        T acc = 0;
                        for (int j = 0; j < ngeom; ++j)
                            acc += coord_dphi[((0 * nd_ + j)) * tdim + k]
                                * cdofs[(side * ngeom + j) * gdim + i];
                        J[i * tdim + k] = acc;
                    }
                md::mdspan<T, md::dextents<std::size_t, 2>> Jm(J.data(), gdim, tdim);
                md::mdspan<T, md::dextents<std::size_t, 2>> Km_m(Km.data(), tdim, gdim);
                if (tdim == gdim)
                    math::inv(Jm, Km_m);
                else
                    math::pinv(Jm, Km_m);

                _map_gradients(pre.test_dphi_ref(local_facet), nq, ndofs0, tdim,
                    gdim, Km, dphi0_phys);
                _map_gradients(pre.trial_dphi_ref(local_facet), nq, ndofs1, tdim,
                    gdim, Km, dphi1_phys);

                // Physical coordinates of the side's points.
                const auto coord_phi = pre.coord_phi(local_facet);
                for (int p = 0; p < nq; ++p)
                    for (int i = 0; i < gdim; ++i) {
                        T acc = 0;
                        for (int j = 0; j < ngeom; ++j)
                            acc += coord_phi[static_cast<std::size_t>(p * ngeom + j)]
                                * cdofs[(side * ngeom + j) * gdim + i];
                        X_phys[static_cast<std::size_t>((side * nq + p)) * gdim + i] = acc;
                    }

                // Coefficient values and gradients of the side.
                for (int c = 0; c < ncoeffs; ++c) {
                    const int offset = pre.coeff_offset(c);
                    const auto cphi_f = pre.coeff_phi(local_facet, c);
                    const std::size_t csize = cphi_f.size() / nq_;
                    for (int p = 0; p < nq; ++p) {
                        T acc = 0;
                        for (std::size_t i = 0; i < csize; ++i)
                            acc += cphi_f[static_cast<std::size_t>(p * csize + i)]
                                * coeffs[(offset + i) + side * pre.coeff_offset(ncoeffs)];
                        coeffs_phys[static_cast<std::size_t>(c * nrows) + side * nq_ + p] = acc;
                    }
                    const auto cdphi_f = pre.coeff_dphi(local_facet, c);
                    const std::size_t csize_ref = cdphi_f.size() / (nq_ * tdim);
                    for (int q = 0; q < gdim; ++q)
                        for (int p = 0; p < nq; ++p) {
                            T gacc = 0;
                            for (int j = 0; j < tdim; ++j)
                                for (std::size_t i = 0; i < csize_ref; ++i)
                                    gacc += cdphi_f[static_cast<std::size_t>((p * csize_ref + i)) * tdim + j]
                                        * Km[j * gdim + q] * coeffs[offset + i + side * pre.coeff_offset(ncoeffs)];
                            dcoeffs_phys[static_cast<std::size_t>((c * nrows + side * nq_ + p)) * gdim + q] = gacc;
                        }
                }
            }

            // Build double-sided data (basis of both sides concatenated).
            std::vector<T> phi0_all(static_cast<std::size_t>(nrows) * ndofs0);
            std::vector<T> phi1_all(static_cast<std::size_t>(nrows) * ndofs1);
            for (int side = 0; side < 2; ++side) {
                const int lf = side == 0 ? local_facet0 : local_facet1;
                const auto p0 = pre.test_phi(lf);
                const auto p1 = pre.trial_phi(lf);
                for (int p = 0; p < nq; ++p) {
                    for (int i = 0; i < ndofs0; ++i)
                        phi0_all[static_cast<std::size_t>((side * nq + p)) * ndofs0 + i]
                            = p0[static_cast<std::size_t>(p * ndofs0 + i)];
                    for (int i = 0; i < ndofs1; ++i)
                        phi1_all[static_cast<std::size_t>((side * nq + p)) * ndofs1 + i]
                            = p1[static_cast<std::size_t>(p * ndofs1 + i)];
                }
            }

            FacetKernelData<T> data;
            data.phi0 = phi0_all;
            data.dphi0 = dphi0_phys;
            data.phi1 = phi1_all;
            data.dphi1 = dphi1_phys;
            data.w = pre.weights();
            data.detJ = detJ;
            data.n = n_phys;
            data.X = X_phys;
            data.coeffs = coeffs_phys;
            data.dcoeffs = dcoeffs_phys;
            data.constants = constants;
            data.num_points = nq;
            data.num_dofs0 = ndofs0;
            data.num_dofs1 = ndofs1;
            data.tdim = tdim;
            data.restricted = 1;
            weak_fn(Ae, data);
        };
    }

    template class FacetPrecomputeData<double>;
    template class FacetPrecomputeData<float>;

    template kernel_t<double> make_facet_kernel(const FacetPrecomputeData<double>&,
        facet_kernel_weak_fn_t<double>);
    template kernel_t<float> make_facet_kernel(const FacetPrecomputeData<float>&,
        facet_kernel_weak_fn_t<float>);

    template kernel_t<double> make_interior_facet_kernel(
        const FacetPrecomputeData<double>&, facet_kernel_weak_fn_t<double>);
    template kernel_t<float> make_interior_facet_kernel(
        const FacetPrecomputeData<float>&, facet_kernel_weak_fn_t<float>);

} // namespace hellofem::fem
