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
    /// The value is the interfaces' own, and a study over several of them
    /// takes the tightest (see `study_time_tolerance`). Measured on this
    /// install (COMSOL 6.2.0.290) by reading the time solver node's own `rtol`
    /// back after a run of a minimal model per set of interfaces, every set
    /// the app can be asked for:
    ///
    ///     interfaces      rtol    time method     max BDF order
    ///     ht              1e-2    BDF             2
    ///     ec              1e-2    BDF             5
    ///     solid           1e-3    generalized-α   5
    ///     ht, ec          1e-2    BDF             2
    ///     ht, solid       1e-3    generalized-α   2
    ///     ec, solid       1e-3    BDF             5
    ///     ec, ht, solid   1e-3    BDF             2
    ///
    /// Each row is the tightest of its interfaces' values, which is what fixes
    /// the three numbers below. The two sets the app's own references were
    /// solved with — heat transfer alone, and the electric-thermal-structural
    /// set — are 1e-2 and 1e-3, the values the cases' own comparisons had
    /// already identified.
    ///
    /// The order column is the reference's other physics-controlled setting: a
    /// maximum BDF order of 2 for every set the app's references run, which is
    /// what `TimeSettings::max_order` carries. A set of the electric and the
    /// structural interface alone would be run at 5, and the app implements
    /// the orders 1 and 2.
    inline constexpr double heat_time_tolerance = 1e-2;
    inline constexpr double electric_time_tolerance = 1e-2;
    inline constexpr double solid_time_tolerance = 1e-3;

    // COMSOL's tolerance levels, and where the app carries each of them.
    //
    // COMSOL nests four tolerances, and a comparison against a reference is
    // only meaningful when the level that is meant is the level that is set.
    // From the outside in:
    //
    // 1. The study step. A time-dependent study's relative tolerance is the
    //    physics interfaces' own unless the study step overrides it
    //    (`usertol`). Those are the three values above, combined by
    //    `study_time_tolerance`.
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
    //    holds every dof of every field to one relative residual instead: the
    //    tolerance *level* above is now the reference's own, but its
    //    per-variable weighting is not.
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
