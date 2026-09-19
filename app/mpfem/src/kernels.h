// hellofem::app — weak-form kernels for the electro-thermal-structural physics
// SPDX-License-Identifier: MIT
#pragma once

#include "fem/facet_precompute.h"
#include "fem/precompute.h"

namespace hellofem::app::kernels {

    using fem::CellKernelData;
    using fem::FacetKernelData;

    // ---- Scalar diffusion: Ae_ij = Σ_q w detJ (∇φi)·(D ∇φj), D = scalar ----
    void diffusion_scalar(double* Ae, const CellKernelData<double>& d);

    // ---- Scalar mass: Ae_ij = Σ_q w detJ c φi φj ----
    void mass_scalar(double* Ae, const CellKernelData<double>& d);

    // ---- Scalar load: Ae_i = Σ_q w detJ f φi ----
    void load_scalar(double* Ae, const CellKernelData<double>& d);

    // ---- Joule heating load (scalar): Ae_i = Σ_q w detJ σ|∇V|² φi ----
    // Coeffs [σ, V]: σ from d.coeffs[0], ∇V from d.dcoeffs[1] (physical
    // coefficient gradient).
    void joule_heat_load(double* Ae, const CellKernelData<double>& d);

    // ---- Convection Robin (facet): Ae_ij = Σ_q w detJ h φi φj ----
    void convection_mass(double* Ae, const FacetKernelData<double>& d);

    // ---- Convection load (facet): Ae_i = Σ_q w detJ h Tinf φi ----
    // Coeffs [h, Tinf]: both from d.coeffs (one block per coefficient).
    void convection_load(double* Ae, const FacetKernelData<double>& d);

    // ---- Thin layer (facet, thermally thin): Ae_ij =
    //      Σ_q w detJ ds k [(∇φi)·(∇φj) - (∇φi·n̂)(∇φj·n̂)]
    // The layer's own tangential conduction, from its thickness `ds` and its
    // conductivity `k`. The gradient is projected onto the facet, so the
    // normal component — which the layer does not conduct — drops out.
    // Coeffs [ds, k]: one block per coefficient.
    void thin_layer_diffusion(double* Ae, const FacetKernelData<double>& d);

    // ---- Linear elasticity (vector, vdim=3): Ae = Σ_q w detJ B^T C B ----
    // Block structure: dof a component i, dof b component j.
    void elasticity(double* Ae, const CellKernelData<double>& d);

    // ---- Thermal expansion load (vector, vdim=3): Ae_i = Σ_q w detJ B^T σ_th ----
    // Coeffs [T, α, E, ν] per point; constants[0] = reference temperature.
    // σ_th (Voigt) = αΔT(2μ+3λ)·[1,1,1,0,0,0] with ΔT = T - T_ref.
    void thermal_expansion_load(double* Ae, const CellKernelData<double>& d);

} // namespace hellofem::app::kernels
