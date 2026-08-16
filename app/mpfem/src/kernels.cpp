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
        out[0] = (c == 0) ? gx : 0.0;              // eps_xx
        out[1] = (c == 1) ? gy : 0.0;              // eps_yy
        out[2] = (c == 2) ? gz : 0.0;              // eps_zz
        out[3] = (c == 2) ? gy : (c == 1) ? gz : 0.0; // gamma_yz
        out[4] = (c == 2) ? gx : (c == 0) ? gz : 0.0; // gamma_xz
        out[5] = (c == 1) ? gx : (c == 0) ? gy : 0.0; // gamma_xy
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
    // Physical dimension from the coefficient-gradient buffer size.
    const int gdim = nq > 0 ? static_cast<int>(d.dcoeffs.size() / (2 * nq)) : 3;
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
        const double hT = d.coeffs[q]; // coeffs[0] = h*Tinf
        for (int i = 0; i < nd; ++i)
            Ae[i] += w * hT * d.phi0[q * nd + i];
    }
}

void elasticity(double* Ae, const CellKernelData<double>& d)
{
    constexpr int vdim = 3;
    const int nq = d.num_points, nd = d.num_dofs0, tdim = d.tdim;
    // Blocked vector element: local dof = scalar_dof * vdim + component.
    const int nds = nd / vdim;
    const int nn = nd * nd;
    std::memset(Ae, 0, nn * sizeof(double));
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
            const double* gA = &d.dphi0[(q * nd + a * vdim) * tdim];
            double BA[3][6]; // BA[c][r] = strain row r for component c
            for (int c = 0; c < vdim; ++c)
                strain_B(BA[c], gA[0], gA[1], gA[2], c);
            for (int b = 0; b < nds; ++b) {
                const double* gB = &d.dphi1[(q * nd + b * vdim) * tdim];
                double BB[3][6];
                for (int c = 0; c < vdim; ++c)
                    strain_B(BB[c], gB[0], gB[1], gB[2], c);
                for (int cb = 0; cb < vdim; ++cb) {
                    // CB[r] = sum_s C[r][s] * BB[cb][s]
                    double CB[6];
                    for (int r = 0; r < 6; ++r) {
                        double acc = 0;
                        for (int s = 0; s < 6; ++s)
                            acc += C[r * 6 + s] * BB[cb][s];
                        CB[r] = acc;
                    }
                    for (int ca = 0; ca < vdim; ++ca) {
                        double acc = 0;
                        for (int r = 0; r < 6; ++r)
                            acc += BA[ca][r] * CB[r];
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
    const double Tref = d.constants ? d.constants[0] : 0.0;
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
