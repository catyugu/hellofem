// hellofem::app — the model defaults of the COMSOL physics the app solves
// SPDX-License-Identifier: MIT
#pragma once

namespace hellofem::app {

    /// COMSOL's reference temperature, 20 °C: the initial temperature of a
    /// heat transfer physics and the reference temperature a thermal
    /// expansion coupling measures the strain from, both of which a model
    /// feature may override.
    inline constexpr double reference_temperature = 293.15;

    /// COMSOL's element order for a physics interface the model does not
    /// give one: quadratic. It is independent of the geometry's order, so a
    /// linear mesh carries a quadratic field unless the model says otherwise.
    inline constexpr int default_element_order = 2;

    /// COMSOL's physics-controlled relative tolerance for a time-dependent
    /// study: what its solver holds the local truncation error of a step to
    /// when the study step leaves the tolerance to the physics interfaces
    /// (`usertol` off, COMSOL's own default, which its generated model scripts
    /// therefore never write). The app's time stepping starts from the same
    /// number, so that a comparison against a COMSOL reference measures the
    /// discretization rather than the difference between two step controllers.
    ///
    /// COMSOL takes the value from the physics interfaces, and it differs
    /// between them: measured on this install, a heat-transfer-only transient
    /// is held to 1e-2, and the electric-thermal-structural set the app solves
    /// to 1e-3 (each reproduced bit for bit by an explicit `rtol` of that
    /// value, and changed by a tolerance an order below it). The app carries
    /// one number, the tighter of the two, so its steps are never coarser than
    /// COMSOL's for the same model; a case that needs another accuracy states
    /// it in its own study step (`rtol`).
    ///
    /// TODO: reproduce COMSOL's tolerance control in full — the value is the
    /// physics interfaces' own (1e-2 for heat transfer alone, 1e-3 for the
    /// three-physics set here), not one app-wide constant, and COMSOL then
    /// scales the error estimate per dependent variable by that variable's own
    /// tolerance factor rather than holding every dof of every field to the
    /// same relative error. A single constant is what the app has; it is not
    /// yet COMSOL's rule.
    inline constexpr double default_time_tolerance = 1e-3;

} // namespace hellofem::app
