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

    // COMSOL's tolerance levels, and where the app carries each of them.
    //
    // COMSOL nests four tolerances, and a comparison against a reference is
    // only meaningful when the level that is meant is the level that is set.
    // From the outside in:
    //
    // 1. The study step. A time-dependent study's relative tolerance is the
    //    physics interfaces' own unless the study step overrides it
    //    (`usertol`). That is `default_time_tolerance` above.
    //
    // 2. The solver node (Stationary Solver / Time-Dependent Solver). Its
    //    "Relative tolerance" is `stol` (default 1e-3). It has two uses: it
    //    terminates a tolerance-based *iterative linear* solve, and it is the
    //    reference error a *direct* solve is checked against when error
    //    checking is on. It is not a residual norm: the nonlinear iteration
    //    terminates on an error *estimate*, not on the residual.
    //
    // 3. The attribute node (Fully Coupled / Segregated). "Tolerance factor"
    //    (`ntolfact`) multiplies stol into the tolerance actually applied
    //    (the app's `AndersonConfig` has no field for it: it applies stol
    //    as-is, which is the factor of 1). "Termination criterion" selects
    //    the estimate the tolerance is compared against — Solution (the
    //    solution-based estimate eU), Residual (the residual-based eL), or
    //    the minimum/maximum of the two. "Residual factor" (`β`, default
    //    1000) scales eL in those last two. The app terminates on the
    //    residual of the frozen system alone, which is the Residual
    //    criterion, so it is the strictest of the four choices — it never
    //    stops on a solution-based estimate that a residual check would have
    //    rejected.
    //
    //    The estimate COMSOL compares is a weighted Euclidean norm over every
    //    dof of every field, with the weight `max(|U_ij|, S_j)` and `S_j` the
    //    scale factor of the dependent variable's scaling method. The app
    //    holds every dof of every field to one relative residual instead, so
    //    it is not yet COMSOL's rule (the same gap the time tolerance notes).
    //
    // 4. The linear solver. Its tolerance is the *operation node's* stol, not
    //    a separate number, and an iterative solve stops when the error
    //    estimate `ρ * ||r||` falls below it, with ρ the "Factor in error
    //    estimate" (default 400) and, under left preconditioning, a separate
    //    `irestol` on the preconditioned residual (default 0.01). The app's
    //    `la::LinearSettings::rtol` (1e-12) and `atol` (1e-14) are therefore
    //    not COMSOL's numbers: they are three orders *tighter* than COMSOL's
    //    stol, and the app's test is relative to the right-hand side where
    //    COMSOL's is relative to the initial residual times ρ. Both
    //    differences make the app stop later, never earlier.
    //
    // A direct solve is not the end of it: COMSOL refines it
    // (`iterrefine` on, at most `maxrefinesteps` = 15 steps) and, inside a
    // nonlinear solve, `nliniterrefine` (off by default). The app's direct
    // solve does no refinement. It does not need one for the systems it
    // factorizes — the app's own residual of a solved solid system measures
    // at 1e-14 relative, so a refinement step has nothing left to remove —
    // but the app's answer is then not COMSOL's *procedure*, only its result.
} // namespace hellofem::app
