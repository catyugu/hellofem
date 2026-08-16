// hellofem::app — weak-form kernels for the electro-thermal-structural physics
// SPDX-License-Identifier: MIT

#include "kernels.h"

#include <cmath>
#include <cstring>

namespace hellofem::app::kernels {
namespace {

    /// Voigt strain-displacement vector B (6 entries) for scalar dof `a`
    /// with physical gradient `g = (gx,gy,gz)` and component `c` (0..2).
    inline void strain_B(double out[6], double gx, double gy, double gz, int c)
    {
        out[0] = (c == 0) ? gx : 0.0; // eps_xx
        out[1] = (c == 1) ? gy : 0.0; // eps_yy
        out[2] = (c == 2) ? gz : 0.0; // eps_zz
        out[3] = (c == 2) ? gz : (c == 1) ? gy : 0.0; // gamma_yz
        out[4] = (c == 2) ? gz : (c == 0) ? gx : 0.0; // gamma_xz
        out[5] = (c == 1) ? gy : (c == 0) ? gx : 0.0; // gamma_xy
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
        const double hT = d.coeffs[q]; // coeffs[0] = h*Tinf
        for (int i = 0; i < nd; ++i)
            Ae[i] += w * hT * d.phi0[q * nd + i];
    }
}

void elasticity(double* Ae, const CellKernelData<double>& d)
{
    constexpr int vdim = 3;
    const int nq = d.num_points, nd = d.num_dofs0, tdim = d.tdim;
    const int nn = (vdim * nd) * (vdim * nd);
    std::memset(Ae, 0, nn * sizeof(double));
    for (int q = 0; q < nq; ++q) {
        const double w = d.w[q] * d.detJ[q];
        const double E = d.coeffs[0 * nq + q];
        const double nu = d.coeffs[1 * nq + q];
        const double lambda = E * nu / ((1 + nu) * (1 - 2 * nu));
        const double mu = E / (2 * (1 + nu));
        double C[36];
        elasticity_C(C, lambda, mu);
        for (int a = 0; a < nd; ++a) {
            const double* gA = &d.dphi0[(q * nd + a) * tdim];
            double BA[6][3]; // BA[i][r] = strain row r for component i
            for (int i = 0; i < vdim; ++i)
                strain_B(BA[i], gA[0], gA[1], gA[2], i);
            for (int b = 0; b < nd; ++b) {
                const double* gB = &d.dphi1[(q * nd + b) * tdim];
                double BB[6][3];
                for (int i = 0; i < vdim; ++i)
                    strain_B(BB[i], gB[0], gB[1], gB[2], i);
                for (int ia = 0; ia < vdim; ++ia) {
                    // CB[r] = sum_s C[r][s] * BB[ib][s]
                    double CB[6];
                    for (int r = 0; r < 6; ++r) {
                        double acc = 0;
                        for (int s = 0; s < 6; ++s)
                            acc += C[r * 6 + s] * BB[ia][s];
                        CB[r] = acc;
                    }
                    for (int ib = 0; ib < vdim; ++ib) {
                        double acc = 0;
                        for (int r = 0; r < 6; ++r)
                            acc += BA[ib][r] * CB[r];
                        Ae[(a * vdim + ib) * (vdim * nd) + (b * vdim + ia)] += w * acc;
                    }
                }
            }
        }
    }
}

void thermal_strain_load(double* Ae, const CellKernelData<double>& d)
{
    constexpr int vdim = 3;
    const int nq = d.num_points, nd = d.num_dofs0, tdim = d.tdim;
    std::memset(Ae, 0, vdim * nd * sizeof(double));
    for (int q = 0; q < nq; ++q) {
        const double w = d.w[q] * d.detJ[q];
        // coeffs: 6 Voigt stress components per point.
        const double* s = &d.coeffs[q * 6];
        for (int a = 0; a < nd; ++a) {
            const double* g = &d.dphi0[(q * nd + a) * tdim];
            double BA[6][3];
            for (int i = 0; i < vdim; ++i)
                strain_B(BA[i], g[0], g[1], g[2], i);
            for (int i = 0; i < vdim; ++i) {
                double acc = 0;
                for (int r = 0; r < 6; ++r)
                    acc += BA[i][r] * s[r];
                Ae[a * vdim + i] += w * acc;
            }
        }
    }
}

} // namespace hellofem::app::kernels
