// hellofem::app — the BDF time discretization and step control of a transient solve
// SPDX-License-Identifier: MIT
#pragma once

#include "defaults.h"
#include "la/Vector.h"

#include <span>
#include <vector>

namespace hellofem::app {

    /// Weights of the time discretization of a first-order-in-time field
    /// equation `M u' + K u = f` at the new level `t_{n+1}`, with the step
    /// `dt`, written as
    ///     Σ_k a_k M u^{n+1-k} + Σ_k b_k K u^{n+1-k} = f(t_{n+1}).
    /// `a` and `b` are already divided by dt and index 0 is the new level.
    /// The BDF formulas are fully implicit, so the load enters at the new
    /// level alone.
    struct TimeWeights {
        std::vector<double> a; // mass-operator weights
        std::vector<double> b; // stiffness-operator weights
    };

    /// One time level of a multistep scheme: the weights that apply at it,
    /// the level time, and the stored data of the previous levels.
    struct TimeLevel {
        TimeWeights weights;
        /// Time of this level (its Dirichlet data is evaluated there).
        double time = 0.0;
        /// Previous solution levels, most recent first.
        std::span<const la::Vector<double>* const> history;
    };

    /// The steps into a level: the step that reached it, and the step before
    /// that. The variable-step weights of order 2 read both, and so does the
    /// truncation error estimate.
    struct TimeSteps {
        double dt = 0.0;
        double dt_previous = 0.0;
    };

    /// How the solver places its steps relative to the output times of the
    /// study: COMSOL's "Steps taken by solver".
    enum class StepsMode {
        /// The step is the solver's own; an output time a step stepped over
        /// is reported by interpolating the scheme's polynomial there.
        free,
        /// The step is the solver's own, but every subinterval of the output
        /// times holds at least one step.
        intermediate,
        /// Every step ends at an output time; the solver takes the steps its
        /// tolerance needs in between.
        strict,
        /// A step of the model's own, with the time-step error test off.
        manual,
    };

    /// Which times the result is stored at: COMSOL's "Times to store".
    enum class StoreMode {
        /// The output times, from the steps that stepped over them.
        interpolate,
        /// The solver step closest to each output time.
        closest,
        /// The solver's own steps.
        steps,
    };

    /// The time stepping of a case: the method, the accuracy its steps are
    /// held to, and where its steps are placed.
    ///
    /// The method is the reference's BDF — the backward differentiation
    /// formulas of order 1 to 5 with the order selected per step. The
    /// reference states the two ends of that selection as "Minimum BDF order"
    /// and "Maximum BDF order", 1 and 2 by default, and those defaults are
    /// what a run takes: the app carries no scheme of its own and its command
    /// line cannot be configured into a different time discretization from
    /// the one the reference used (see the order selection of `BdfController`
    /// and the measurements recorded there).
    ///
    /// The reference's other implicit method, generalized-α, is not
    /// implemented: it is a one-step method on the *coupled* system, in which
    /// every variable is advanced by the formula its own time order calls for
    /// (first-order in time for heat and electric potential, second-order for
    /// a structural field with inertia). The app advances one field at a time
    /// over a shared level, so it can express a multistep method on each field
    /// and not a coupled one-step method. Nothing of the app runs a
    /// generalized-α transient, and the reference's structural default — which
    /// does use it, with inertia — is the case that would need it.
    struct TimeSettings {
        /// Lowest order the BDF order selection may take (the reference's
        /// "Minimum BDF order", 1 by default: it exists to keep a solver off
        /// the first order, and at its default it holds nothing back).
        int min_order = 1;

        /// Highest order the selection may take, and the order a `manual`
        /// step is taken at (the reference's "Maximum BDF order" and its
        /// "BDF order" of a manual step, both 2 by default — and 2 is the
        /// highest the app implements: the orders above it need the
        /// variable-step weights of a longer history, and no reference of the
        /// app's is run at them).
        int max_order = 2;

        /// Where the steps are placed relative to the output times.
        StepsMode steps = StepsMode::free;

        /// Which times are stored.
        StoreMode store = StoreMode::interpolate;

        /// Whether the stepping may pass the last output time, whose value is
        /// then interpolated from the steps around it. With it off, the
        /// stepping stops at the last time and takes no step past it. The
        /// reference has it on for every setting of "Steps taken by solver"
        /// except `strict`, which reaches the last time by its own rule and
        /// needs nothing of this.
        bool interpolate_end_time = true;

        /// The relative tolerance `R`: the local truncation error of a step is
        /// held to `A + R |u_i|` per dof, and the step meets the tolerance
        /// while the weighted RMS norm of those ratios is at most one. It
        /// bounds the error of a step, so it is the accuracy of the time
        /// discretization — not of the linear or the nonlinear solver. A study
        /// takes the value the reference's own physics-controlled tolerance
        /// has for its physics (`study_time_tolerance`), so that a run measures
        /// the discretization rather than the difference between two step
        /// controllers, and a model that states its own tolerance replaces it.
        /// The default is the tightest of the interfaces' values, which is
        /// what a `TimeSettings` built without a model is held to.
        double tolerance = solid_time_tolerance;

        /// The reference's absolute tolerance, which is `A = R *
        /// absolute_factor` under its `Factor` method (its default method, of
        /// factor 0.1). It is what keeps a dof whose value is near zero —
        /// round-off around a clamped face, a field that starts at zero —
        /// from being held to a relative accuracy on a value that carries
        /// nothing, and it is the whole of the absolute part of the error
        /// weight (the reference's `Manual` method instead states `A` itself,
        /// 0.001 by default; nothing of the app's sets one).
        double absolute_factor = 0.1;

        /// The step of `StepsMode::manual`, in seconds.
        double manual_step = 0.0;
    };

    /// The weights of a BDF step of `order` (1 or 2) over the steps
    /// `steps`, with `h = steps.dt` and `h' = steps.dt_previous`:
    ///
    ///     order 1:  u' = (u^{n+1} - u^n) / h
    ///     order 2:  u' = the derivative at t_{n+1} of the quadratic through
    ///               u^{n-1}, u^n, u^{n+1}, i.e.
    ///               a = {(2h + h') / (h (h + h')), -(h + h') / (h h'),
    ///                    h / (h' (h + h'))}
    ///
    /// which is (3/2, -2, 1/2) / h when the two steps are equal, and stays
    /// second order for any step ratio: the weights of a variable step are
    /// what keeps a BDF2 run second order when the step size changes.
    TimeWeights bdf_weights(int order, TimeSteps steps);

    /// The one time level of a steady problem: the degenerate one-level
    /// scheme `K u = f` of the same `M u' + K u = f` a transient assembles a
    /// step of, so a physics states its steady system through the same
    /// `assemble_step` it states a time step through.
    TimeLevel steady_level(double t);

    /// The first step of a run, as a fraction of the span it integrates: the
    /// reference takes it below this, and its own log shows the value itself
    /// (0.6 s over its 600 s span), so the derivative condition of
    /// `BdfController::first_step` is what may take it lower still.
    inline constexpr double first_step_fraction = 1e-3;

    /// The smallest step of a run, as a fraction of the span it integrates:
    /// below this the error cannot be met by shrinking the step any more, and
    /// the interval is one the estimate cannot resolve.
    inline constexpr double smallest_step_fraction = 1e-10;

    /// The largest step of a run, as a fraction of the span it integrates:
    /// the reference's "Maximum step constraint" at its automatic setting.
    /// Measured on the reference: COMSOL's steps for EcTSmBusbarTransient
    /// stop growing at 60 s over a 600 s span and stay there.
    inline constexpr double max_step_fraction = 0.1;

    /// The artificial backward-Euler step of consistent initialization, as a
    /// fraction of the initial step: the reference takes one before its BDF
    /// stepping begins, to reconcile the initial values with the constraints
    /// (its log shows it as the 0.0012 s the first step starts from). Its
    /// default is 0.001.
    inline constexpr double backward_euler_step_fraction = 1e-3;

    /// The safety factor of that initialization, used in the algebraic
    /// termination of the step: a larger value is stricter, and the default
    /// of 20 "corresponds to the normal behavior for any new time step". The
    /// app's level solves already terminate on the residual of the frozen
    /// system, which is the strictest of the reference's own criteria (see
    /// `defaults.h`), so the factor has nothing left to tighten and is
    /// recorded here rather than applied.
    inline constexpr double backward_euler_safety_factor = 20.0;

    /// The accuracy with which the reference's solver lands on an event: the
    /// tolerance of the root finding that locates an implicit event's
    /// condition in time, so a smaller value resolves the crossing more
    /// tightly (the reference's own guidance for overshooting an event is to
    /// tighten it). It applies to the root finding alone and not to the
    /// startup phase of the stepping.
    ///
    /// The app implements no events: an event is a feature of the model — an
    /// explicit one states the times it triggers at, an implicit one states a
    /// condition on the solution and an indicator whose sign change triggers
    /// it — and it reinitializes the solution (and any discrete states) when
    /// it fires, storing the solution before and after. Nothing of the app's
    /// reads or writes such a feature, so there is no root to find and this
    /// tolerance has nothing to bound. It is recorded as the mechanism it is,
    /// not as a setting.
    inline constexpr double event_tolerance = 0.01;

    /// The local truncation error of the step just taken, at the orders the
    /// order selection compares: the reference solver's own `err_k`,
    /// `err_{k-1}` and `err_{k+1}`, with `k` the order the step was taken at.
    ///
    /// They are its `err_j = sigma[j] ||ee||`, where `ee` is the Newton
    /// correction of the step — the difference between the corrected solution
    /// and the value the degree-`j` polynomial through the last levels
    /// extrapolates to it. At a constant step `sigma[j] = 1/(j+1)` and `ee` is
    /// the `(j+1)`-th backward difference of the computed levels, so
    ///
    ///     err_j = j! |dt^(j+1) DD_(j+1) u| = |∇^(j+1) u| / (j + 1)
    ///
    /// in the scaled divided differences `d_(j+1) = dt^(j+1) DD_(j+1) u` the
    /// stepper already forms. That is NOT the textbook local truncation error
    /// `c_j d_(j+1)`, whose coefficient is 1/2 at the order 1 and 2/9 at the
    /// order 2: the correction carries the polynomial's extrapolation
    /// remainder as well as the corrector's own truncation error, and at the
    /// order 2 the two differ by a factor of 2.5 — which is enough to decide
    /// the order differently.
    ///
    /// An estimate the level history does not reach is reported as zero, which
    /// the selection reads as "not available". A step meets the tolerance
    /// while `at` is at most one.
    struct StepError {
        double below = 0.0;
        double at = 0.0;
        double above = 0.0;
    };

    /// The step size and order control of the BDF method: the algorithm of
    /// the solver the reference runs, IDA (LLNL), which is what its
    /// Time-Dependent Solver uses for BDF. The reference's own solver log is
    /// what the rules below are read from and checked against.
    ///
    /// - **Startup.** Until the order reaches its maximum, or a step fails,
    ///   or the order is lowered, every step is taken one order higher and at
    ///   twice the size: there is no history to select from yet. The
    ///   reference's sequence over its 600 s span — steps of 0.6, 0.6, 1.2,
    ///   2.4, 4.8, 9.6, 19.2, 19.2, 38.4, 38.4, 60, ... s at orders 1, 1, 2,
    ///   1, 1, ... — is this phase (the first two steps) followed by the
    ///   steady one.
    /// - **The deadbeat region.** A step that met the tolerance is *not*
    ///   grown by the asymptotic formula. The factor is the clamp of
    ///   `(2 error + 1e-4)^(-1/(q+1))` to [0.5, 0.9] below one, one inside
    ///   the band (1, 2), and 2 above it — so a step doubles only while its
    ///   estimate sits below `2^-(q+2)` of the tolerance (1/8 at order 1,
    ///   1/16 at order 2, which is the "16 times smaller" the reference's own
    ///   documentation describes as the deadbeat region) and is held at its
    ///   size otherwise. The two halves of that sentence are the same rule:
    ///   the "deadbeat region" is the band in which the step is *held*, and
    ///   the factor of two is what it grows by once the estimate leaves it.
    ///   A controller that grows a step by the asymptotic factor from inside
    ///   the band doubles the number of steps it takes (measured on the
    ///   reference's case: 79 steps against 18).
    /// - **A rejected step** is retried at
    ///   `0.9 (2 error + 1e-4)^(-1/(q+1))` of its size, at most 0.9 and at
    ///   least 0.25 of it, at the order the estimate favours; a second
    ///   rejection takes a quarter of the step, and a third one the order 1.
    /// - **The order** is reconsidered only after `q + 2` steps at a constant
    ///   order `q` and a constant step size, and not on the step after a
    ///   change of order. The estimates it reads are the reference's own (see
    ///   `StepError`), compared as the truncation error norms `terr_j =
    ///   (j + 1) err_j`: the order 1 is raised to 2 when `terr_2` is below half
    ///   of `terr_1`, and the order 2 is lowered when `terr_1` is at least
    ///   twice as small as `terr_2`. The reference's own rule at the orders
    ///   above 2 reads one estimate further back than the history keeps, and
    ///   those orders are refused (see `TimeSettings`).
    ///
    /// One controller drives the whole case: the step size and the order are
    /// the study's, and every field is advanced at them.
    class BdfController {
    public:
        /// @param[in] settings the method's settings: its order range, its
        /// step and store modes and its tolerance.
        /// @param[in] span the span the study integrates, from which the first
        /// step and the largest one are taken.
        BdfController(const TimeSettings& settings, double span);

        /// The first step of the run, from the weighted norm
        /// `derivative_norm` of the time derivative the initial state starts
        /// with: at most `first_step_fraction` of the span, and at most
        /// `1/2 / derivative_norm`, the derivative condition the reference
        /// states as `DT yp_norm <= 1/2`.
        double first_step(double derivative_norm) const;

        /// Begin the run at the step `h`, the state being at its first level.
        void start(double h);

        /// The step size of the next step.
        double step_size() const;

        /// The order the next step is taken at.
        int order() const;

        /// Whether the local truncation error governs the step. It does not
        /// in `StepsMode::manual`, where the step is the model's own and only
        /// the algebraic error of each level's solve still applies.
        bool error_controlled() const;

        /// The smallest step the controller will propose, below which the
        /// error cannot be met on this interval at all.
        double smallest_step() const { return smallest_; }

        /// Accept the step of size `h` just taken, whose error estimates are
        /// `error`: select the size and the order of the next one.
        void accept(const StepError& error, double h);

        /// Drop the step of size `h` just taken: it missed the tolerance, and
        /// the next attempt is smaller.
        void reject(const StepError& error, double h);

    private:
        /// The order the error test suggests taking the retried step at: the
        /// one below, when the estimate there is at least twice as small.
        int lowered_order(const StepError& error) const;

        /// The factor the step is multiplied by after a step whose estimate
        /// was `error` (see the deadbeat region above).
        double factor(double error, int order) const;

        TimeSettings settings_;
        double span_ = 0.0;
        double largest_ = 0.0;
        double smallest_ = 0.0;
        /// The step proposed for the next step, and the one the last step
        /// was taken at.
        double h_ = 0.0;
        double used_ = 0.0;
        /// The order of the next step, and of the last one.
        int order_ = 1;
        int order_used_ = 0;
        /// The steps taken at this step size and order, and the steps
        /// accepted over the run.
        int ns_ = 0;
        int steps_ = 0;
        /// The failures of the step being attempted.
        int failures_ = 0;
        /// Whether the startup phase is still running.
        bool startup_ = true;
    };

    /// The k-th Newton divided difference of the solution levels, scaled by
    /// the step into the newest level: `dt^k DD_k u`, per dof. `levels` are
    /// the solutions, most recent first at least the new one, at the times
    /// `times`; a set too short for the order, i.e. fewer than k + 1 levels,
    /// gives zeros.
    la::Vector<double> divided_difference(int k,
        std::span<const la::Vector<double>* const> levels,
        std::span<const double> times);

    /// The value at time `t` of the polynomial through `levels`, whose times
    /// are `times` (newest first). This is the interpolation a transient
    /// result is reported with at an output time the steps stepped over: the
    /// polynomial is the Newton form of the levels, so it reproduces them
    /// exactly at their own times and its degree is one less than their count.
    la::Vector<double> interpolate_levels(
        std::span<const la::Vector<double>* const> levels,
        std::span<const double> times, double t);

} // namespace hellofem::app
