// hellofem::app — time stepping schemes and step-size control of a transient solve
// SPDX-License-Identifier: MIT
#pragma once

#include "defaults.h"
#include "la/Vector.h"

#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace hellofem::app {

    /// Weights of the time discretization of a first-order-in-time field
    /// equation `M u' + K u = f` at the new level `t_{n+1}`, with the step
    /// `dt`, written as
    ///     Σ_k a_k M u^{n+1-k} + Σ_k b_k K u^{n+1-k}
    ///         = c_new f(t_{n+1}) + c_old f(t_n).
    /// `a` and `b` are already divided by dt and index 0 is the new level;
    /// the schemes implemented here need the source at two levels only.
    struct TimeWeights {
        std::vector<double> a; // mass-operator weights
        std::vector<double> b; // stiffness-operator weights
        double c_new = 1.0;
        double c_old = 0.0;
    };

    /// One time level of a multistep scheme: the weights that apply at it,
    /// the level time, and the stored data of the previous levels.
    struct TimeLevel {
        TimeWeights weights;
        /// Time of this level (its Dirichlet data is evaluated there).
        double time = 0.0;
        /// Previous solution levels, most recent first.
        std::span<const la::Vector<double>* const> history;
        /// Source load of the previous level (null at the first step).
        const la::Vector<double>* source_old = nullptr;
        /// Source load of this level; stored for the next step.
        la::Vector<double>* source_new = nullptr;
    };

    /// The steps into a level: the step that reached it, and the step before
    /// that. The variable-step weights of order 2 read both, and so does the
    /// truncation error estimate.
    struct TimeSteps {
        double dt = 0.0;
        double dt_previous = 0.0;
    };

    /// What a family of schemes derives its weights from.
    enum class TimeFamily {
        /// Backward differentiation: the time derivative at the new level is
        /// read off the polynomial through the last levels, so the weights
        /// follow from the steps (and sum to zero, which makes a constant
        /// state have no time derivative).
        bdf,
        /// Crank-Nicolson: the trapezoidal average of the two levels, second
        /// order at a constant step.
        crank_nicolson,
    };

    /// A time stepping scheme: its family, and the order it is taken at.
    struct TimeScheme {
        /// Name the `--scheme` option accepts.
        std::string_view name;
        TimeFamily family = TimeFamily::bdf;
        /// Order of accuracy in dt at a constant step: the order the scheme
        /// is taken at, or, for the BDF family, the highest order the driver
        /// may select for it.
        int order = 1;
    };

    /// The time stepping of a case: the discretization, and the accuracy its
    /// steps are held to.
    ///
    /// The scheme says how a level is discretized: its family, and the order
    /// it is taken at — or, for the BDF family, the highest order the driver
    /// may select. How far apart the levels are is not the scheme's: the step
    /// size and the order of every step follow from the local truncation
    /// error, and the output times are reported by interpolating the levels
    /// the steps produced (see `CaseScheduler::advance_adaptive`).
    struct TimeSettings {
        /// The scheme, as `--scheme` names it.
        ///
        /// The app's default is the BDF family at the order the reference's own
        /// solver selected, so that a transient comparison measures the two
        /// models rather than the two time discretizations. COMSOL selects that
        /// order itself (its Time-Dependent Solver has "Maximum BDF order" 2 and
        /// "Minimum BDF order" 1, both its own defaults), and its log shows what
        /// it selects: over the 18 steps of EcTSmBusbarTransient the "Order"
        /// column reads 1,1,2,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2 — fourteen of the
        /// eighteen at order 1, the two excursions to order 2 single steps at
        /// either end.
        ///
        /// Measured against that reference, the order is what the comparison
        /// turns on: order 1 reproduces it to 2.96e-05 (L2 relative, the
        /// temperature, worst over the output times) where order 2 sits at
        /// 2.33e-04, outside the case's tolerance on 15 of the 63 exported
        /// columns against order 1's none. On TBlockTransient and
        /// TBlockNonlinTransient the two orders are indistinguishable (every
        /// column passes at either).
        std::string scheme = "bdf1";
        /// The relative tolerance the truncation error of a step is held to,
        /// weighted per dof by `tolerance max(|u|, scale)` (see
        /// `TimeStepper::error`). It bounds the error of a step, so it is the
        /// accuracy of the time discretization — not of the linear or the
        /// nonlinear solver. It starts at the value COMSOL's own
        /// physics-controlled tolerance takes for the physics the app solves,
        /// so that a run measures the discretization rather than the
        /// difference between two step controllers; a model that states its
        /// own tolerance replaces it, and the command line replaces both.
        double tolerance = default_time_tolerance;
    };

    /// The available schemes.
    std::span<const TimeScheme> time_schemes();

    /// The scheme named `name`, case-insensitive. Throws for an unknown name.
    const TimeScheme& find_time_scheme(std::string_view name);

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

    /// The leading coefficient of the local truncation error of a step of
    /// `order` of `family`, against the divided difference it is estimated
    /// from: that error is `C dt^(order+1) u^(order+1)`, and
    /// `u^(order+1) = (order+1)! DD_(order+1)`, so the estimate the step is
    /// controlled by is `c dt^(order+1) DD_(order+1)` with
    /// `c = (order+1)! C`. The constants are the textbook ones: 1/2 for
    /// backward Euler, 2/9 for BDF2, 1/12 for the trapezoidal rule.
    double error_coefficient(TimeFamily family, int order);

    /// The Crank-Nicolson weights of a step of `dt`: the trapezoidal average
    /// of the stiffness and of the load over the two levels, which is what
    /// makes it second order.
    TimeWeights cn_weights(double dt);

    /// The first step of a run, as a fraction of the span it integrates: a
    /// first step has no history for the error estimate to read, so it has to
    /// be small enough that its own error cannot escape the tolerance, and
    /// the controller raises it within a few steps wherever the solution
    /// allows.
    inline constexpr double first_step_fraction = 1e-3;

    /// The smallest step of a run, as a fraction of the span it integrates:
    /// below this the error cannot be met by shrinking the step any more, and
    /// the interval is one the estimate cannot resolve.
    inline constexpr double smallest_step_fraction = 1e-10;

    /// The factor the next step is multiplied by after a step whose local
    /// error measured `error` against the tolerance (1 = exactly at it),
    /// following the controller of the COMSOL BDF solver:
    ///
    ///     h_new = clamp(0.9 error^(-1/(order+1)), 0.1, 2) h
    ///
    /// the asymptotic dependence of the local truncation error on the step,
    /// with a safety factor. Every step proposes with it, the ones that met the
    /// tolerance included, so a step whose estimate sits below 0.9^(order+1) of
    /// the tolerance grows and one above it shrinks.
    ///
    /// Measured on the reference: COMSOL's own step sequence for
    /// EcTSmBusbarTransient doubles from the first step to the cap in eight
    /// steps (0.6, 0.6, 1.2, 2.4, 4.8, 9.6, 19.2, 19.2, 38.4, 38.4, 60, 60,
    /// ...), which is the clamp of 2 taken at every one of them — a controller
    /// that only grew a step once its estimate was far inside the tolerance
    /// would not reach the cap in that many.
    double step_factor(double error, int order);

    /// The largest step of a run, as a fraction of the span it integrates.
    /// Measured on the reference: COMSOL's steps for EcTSmBusbarTransient stop
    /// growing at 60 s over a 600 s span, and stay there.
    inline constexpr double max_step_fraction = 0.1;

    /// The order the next step is taken at — the BDF family's, a scheme of a
    /// fixed order keeping its own — from the scaled derivative norms
    /// `derivative_scale[k - 1] = |dt^k DD_k u|`, DD being the Newton divided
    /// difference and the levels the last ones: the terms of the Taylor
    /// expansion of the solution in the step. The higher order is worth
    /// taking while those terms keep shrinking (a smooth solution resolves
    /// against the higher order better), and the lower one is what a solution
    /// whose terms do not shrink can be resolved at.
    int next_order(int order, int max_order, std::span<const double> derivative_scale);

    /// The scaled derivative norms `derivative_scale[k - 1] = |dt^k DD_k u|`
    /// of the solution levels `levels` (most recent first, that one being the
    /// new level) at the times `times`: the terms of the Taylor expansion of
    /// the solution in the step, maximized over the dofs. One entry per order
    /// the level set supports — a set of n levels has n - 1 of them, so a
    /// short one simply lacks the higher orders.
    std::vector<double> derivative_scale(
        std::span<const la::Vector<double>* const> levels,
        std::span<const double> times);

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
