// hellofem::app — weak-form kernels for the electro-thermal-structural physics
// SPDX-License-Identifier: MIT
#pragma once

#include "fem/precompute.h"
#include "fem/facet_precompute.h"

namespace hellofem::app::kernels {

    using fem::CellKernelData;
    using fem::FacetKernelData;

    // ---- Scalar diffusion: Ae_ij = Σ_q w detJ (∇φi)·(D ∇φj), D = scalar ----
    void diffusion_scalar(double* Ae, const CellKernelData<double>& d);

    // ---- Scalar mass: Ae_ij = Σ_q w detJ c φi φj ----
    void mass_scalar(double* Ae, const CellKernelData<double>& d);

    // ---- Scalar load: Ae_i = Σ_q w detJ f φi ----
    void load_scalar(double* Ae, const CellKernelData<double>& d);

    // ---- Convection Robin (facet): Ae_ij = Σ_q w detJ h φi φj ----
    void convection_mass(double* Ae, const FacetKernelData<double>& d);

    // ---- Convection load (facet): Ae_i = Σ_q w detJ h Tinf φi ----
    void convection_load(double* Ae, const FacetKernelData<double>& d);

    // ---- Linear elasticity (vector, vdim=3): Ae = Σ_q w detJ B^T C B ----
    // Block structure: dof a component i, dof b component j.
    void elasticity(double* Ae, const CellKernelData<double>& d);

    // ---- Thermal strain load (vector, vdim=3): Ae_i = Σ_q w detJ B^T σ_th ----
    // d.coeffs carries the 6 Voigt stress components per point.
    void thermal_strain_load(double* Ae, const CellKernelData<double>& d);

} // namespace hellofem::app::kernels
