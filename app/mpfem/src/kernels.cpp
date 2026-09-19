// hellofem::app — weak-form kernels for the electro-thermal-structural physics
// SPDX-License-Identifier: MIT

#include "kernels.h"

#include <cmath>
#include <cstring>

namespace hellofem::app::kernels {
    namespace {

        /// Voigt strain-displacement vector B (6 entries) for scalar dof `a`
        /// with physical gradient `g = (gx,gy,gz)` and component `c` (0..2).
        /// The displacement is `u_c = φ`, others zero, so the shear terms pick
        /// the gradient of the OTHER index: e.g. gamma_xy = ∂u_x/∂y + ∂u_y/∂x.
        inline void strain_B(double out[6], double gx, double gy, double gz, int c)
        {
            out[0] = (c == 0) ? gx : 0.0; // eps_xx
            out[1] = (c == 1) ? gy : 0.0; // eps_yy
            out[2] = (c == 2) ? gz : 0.0; // eps_zz
            out[3] = (c == 2) ? gy : (c == 1) ? gz
                                              : 0.0; // gamma_yz
            out[4] = (c == 2) ? gx : (c == 0) ? gz
                                              : 0.0; // gamma_xz
            out[5] = (c == 1) ? gx : (c == 0) ? gy
                                              : 0.0; // gamma_xy
        }

        /// Voigt (6x6) elasticity tensor from Lamé parameters.
        inline void elasticity_C(double C[36], double lambda, double mu)
        {
            std::memset(C, 0, 36 * sizeof(double));
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    C[i * 6 + j] = (i == j) ? lambda + 2 * mu : lambda;
            C[3 * 6 + 3] = C[4 * 6 + 4] = C[5 * 6 + 5] = mu;
        }

    } // namespace

    void diffusion_scalar(double* Ae, const CellKernelData<double>& d)
    {
        const int nq = d.num_points, nd = d.num_dofs0, tdim = d.tdim;
        const int nn = nd * nd;
        std::memset(Ae, 0, nn * sizeof(double));
        for (int q = 0; q < nq; ++q) {
            const double w = d.w[q] * d.detJ[q];
            const double D = d.coeffs[q]; // scalar coefficient
            for (int i = 0; i < nd; ++i)
                for (int j = 0; j < nd; ++j) {
                    double dot = 0;
                    for (int c = 0; c < tdim; ++c)
                        dot += d.dphi0[(q * nd + i) * tdim + c]
                            * d.dphi1[(q * nd + j) * tdim + c];
                    Ae[i * nd + j] += w * D * dot;
                }
        }
    }

    void mass_scalar(double* Ae, const CellKernelData<double>& d)
    {
        const int nq = d.num_points, nd = d.num_dofs0;
        std::memset(Ae, 0, nd * nd * sizeof(double));
        for (int q = 0; q < nq; ++q) {
            const double w = d.w[q] * d.detJ[q];
            const double c = d.coeffs[q];
            for (int i = 0; i < nd; ++i)
                for (int j = 0; j < nd; ++j)
                    Ae[i * nd + j] += w * c * d.phi0[q * nd + i] * d.phi1[q * nd + j];
        }
    }

    void load_scalar(double* Ae, const CellKernelData<double>& d)
    {
        const int nq = d.num_points, nd = d.num_dofs0;
        std::memset(Ae, 0, nd * sizeof(double));
        for (int q = 0; q < nq; ++q) {
            const double w = d.w[q] * d.detJ[q];
            const double f = d.coeffs[q];
            for (int i = 0; i < nd; ++i)
                Ae[i] += w * f * d.phi0[q * nd + i];
        }
    }

    void joule_heat_load(double* Ae, const CellKernelData<double>& d)
    {
        const int nq = d.num_points, nd = d.num_dofs0;
        // The coefficient gradients hold the physical dimension the
        // precomputed data is built for (see `CellKernelData::dcoeffs`).
        constexpr int gdim = fem::PrecomputeData<double>::gdim();
        std::memset(Ae, 0, nd * sizeof(double));
        for (int q = 0; q < nq; ++q) {
            const double w = d.w[q] * d.detJ[q];
            const double sigma = d.coeffs[q]; // coeffs[0] = σ
            const double* gV = &d.dcoeffs[(nq + q) * gdim]; // coeffs[1] = ∇V
            double j2 = 0;
            for (int k = 0; k < gdim; ++k)
                j2 += gV[k] * gV[k];
            for (int i = 0; i < nd; ++i)
                Ae[i] += w * sigma * j2 * d.phi0[q * nd + i];
        }
    }

    void convection_mass(double* Ae, const FacetKernelData<double>& d)
    {
        const int nq = d.num_points, nd = d.num_dofs0;
        std::memset(Ae, 0, nd * nd * sizeof(double));
        for (int q = 0; q < nq; ++q) {
            const double w = d.w[q] * d.detJ[q];
            const double h = d.coeffs[q];
            for (int i = 0; i < nd; ++i)
                for (int j = 0; j < nd; ++j)
                    Ae[i * nd + j] += w * h * d.phi0[q * nd + i] * d.phi1[q * nd + j];
        }
    }

    void convection_load(double* Ae, const FacetKernelData<double>& d)
    {
        const int nq = d.num_points, nd = d.num_dofs0;
        std::memset(Ae, 0, nd * sizeof(double));
        for (int q = 0; q < nq; ++q) {
            const double w = d.w[q] * d.detJ[q];
            // Coeffs [h, Tinf].
            const double hT = d.coeffs[q] * d.coeffs[nq + q];
            for (int i = 0; i < nd; ++i)
                Ae[i] += w * hT * d.phi0[q * nd + i];
        }
    }

    void thin_layer_diffusion(double* Ae, const FacetKernelData<double>& d)
    {
        // The facet normal always carries three components (the geometry is
        // three-dimensional), while the gradients carry `tdim`.
        constexpr int gdim = 3;
        const int nq = d.num_points, nd = d.num_dofs0, tdim = d.tdim;
        std::memset(Ae, 0, nd * nd * sizeof(double));
        for (int q = 0; q < nq; ++q) {
            const double w = d.w[q] * d.detJ[q];
            const double ds_k = d.coeffs[q] * d.coeffs[nq + q];
            // `n` is the scaled normal: dividing by detJ recovers the unit
            // one, so the projected gradient below needs the same division.
            double nhat[3];
            for (int c = 0; c < tdim; ++c)
                nhat[c] = d.n[q * gdim + c] / d.detJ[q];
            for (int i = 0; i < nd; ++i) {
                const double* gi = &d.dphi0[(q * nd + i) * tdim];
                double gi_n = 0;
                for (int c = 0; c < tdim; ++c)
                    gi_n += gi[c] * nhat[c];
                for (int j = 0; j < nd; ++j) {
                    const double* gj = &d.dphi1[(q * nd + j) * tdim];
                    double gij = 0, gj_n = 0;
                    for (int c = 0; c < tdim; ++c) {
                        gij += gi[c] * gj[c];
                        gj_n += gj[c] * nhat[c];
                    }
                    Ae[i * nd + j] += w * ds_k * (gij - gi_n * gj_n);
                }
            }
        }
    }

    void elasticity(double* Ae, const CellKernelData<double>& d)
    {
        constexpr int vdim = 3;
        // Scalar dofs of a cell of the app's heaviest element: a cubic
        // Lagrange hexahedron. The strain rows below are formed on the stack.
        constexpr int max_scalar_dofs = 64;
        const int nq = d.num_points, nd = d.num_dofs0, tdim = d.tdim;
        // Blocked vector element: local dof = scalar_dof * vdim + component.
        const int nds = nd / vdim;
        const int nn = nd * nd;
        std::memset(Ae, 0, nn * sizeof(double));

        // Strain rows of the test and trial dofs and the trial's C : B. None
        // of them depends on the other side's dof, so each is formed once per
        // quadrature point and the element tensor only pairs them.
        double BA[max_scalar_dofs][vdim][6];
        double CB[max_scalar_dofs][vdim][6];
        for (int q = 0; q < nq; ++q) {
            const double w = d.w[q] * d.detJ[q];
            const double E = d.coeffs[0 * nq + q];
            const double nu = d.coeffs[1 * nq + q];
            const double lambda = E * nu / ((1 + nu) * (1 - 2 * nu));
            const double mu = E / (2 * (1 + nu));
            double C[36];
            elasticity_C(C, lambda, mu);

            for (int a = 0; a < nds; ++a) {
                // Scalar basis function a is identical across components.
                const double* g = &d.dphi0[(q * nd + a * vdim) * tdim];
                for (int c = 0; c < vdim; ++c)
                    strain_B(BA[a][c], g[0], g[1], g[2], c);
            }
            for (int b = 0; b < nds; ++b) {
                const double* g = &d.dphi1[(q * nd + b * vdim) * tdim];
                double BB[vdim][6];
                for (int c = 0; c < vdim; ++c) {
                    strain_B(BB[c], g[0], g[1], g[2], c);
                    // CB[c][r] = sum_s C[r][s] * BB[c][s]
                    for (int r = 0; r < 6; ++r) {
                        double acc = 0;
                        for (int s = 0; s < 6; ++s)
                            acc += C[r * 6 + s] * BB[c][s];
                        CB[b][c][r] = acc;
                    }
                }
            }

            for (int a = 0; a < nds; ++a) {
                for (int ca = 0; ca < vdim; ++ca) {
                    const double* bA = BA[a][ca];
                    for (int b = 0; b < nds; ++b) {
                        for (int cb = 0; cb < vdim; ++cb) {
                            const double* cB = CB[b][cb];
                            double acc = 0;
                            for (int r = 0; r < 6; ++r)
                                acc += bA[r] * cB[r];
                            Ae[(a * vdim + ca) * nd + (b * vdim + cb)] += w * acc;
                        }
                    }
                }
            }
        }
    }

    void thermal_expansion_load(double* Ae, const CellKernelData<double>& d)
    {
        constexpr int vdim = 3;
        const int nq = d.num_points, nd = d.num_dofs0, tdim = d.tdim;
        // Blocked vector element: local dof = scalar_dof * vdim + component.
        const int nds = nd / vdim;
        const double Tref = d.constants[0];
        std::memset(Ae, 0, nd * sizeof(double));
        for (int q = 0; q < nq; ++q) {
            const double w = d.w[q] * d.detJ[q];
            // Coeffs [T, α, E, ν] per point; σ_th = αΔT(2μ+3λ) diagonal.
            const double dT = d.coeffs[q] - Tref;
            const double alpha = d.coeffs[nq + q];
            const double E = d.coeffs[2 * nq + q];
            const double nu = d.coeffs[3 * nq + q];
            const double lambda = E * nu / ((1 + nu) * (1 - 2 * nu));
            const double mu = E / (2 * (1 + nu));
            const double st = alpha * dT * (2 * mu + 3 * lambda);
            const double sth[6] = {st, st, st, 0, 0, 0};
            for (int a = 0; a < nds; ++a) {
                const double* g = &d.dphi0[(q * nd + a * vdim) * tdim];
                double BA[3][6];
                for (int c = 0; c < vdim; ++c)
                    strain_B(BA[c], g[0], g[1], g[2], c);
                for (int c = 0; c < vdim; ++c) {
                    double acc = 0;
                    for (int r = 0; r < 6; ++r)
                        acc += BA[c][r] * sth[r];
                    Ae[a * vdim + c] += w * acc;
                }
            }
        }
    }

} // namespace hellofem::app::kernels
