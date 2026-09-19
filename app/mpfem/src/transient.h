// hellofem::app — time stepping driver of a time-dependent field
// SPDX-License-Identifier: MIT
#pragma once

#include "field.h"
#include "time_scheme.h"

#include <cstddef>
#include <vector>

namespace hellofem::app {

    /// Advances a time-dependent field with the BDF scheme.
    ///
    /// Owns the solution history, which the scheme and the error estimate
    /// read, and refreshes the field at every new level. The solution of the
    /// previous level is the initial guess of the next one — for the nonlinear
    /// material update and for the Krylov solve.
    ///
    /// The step size and the order are not the stepper's: they are the case's,
    /// selected by `BdfController` from the error estimates of every field of
    /// the level at once. A step is kept until the driver either accepts it
    /// (the next step is taken from it) or drops it with `undo`.
    class TimeStepper {
    public:
        /// Advance `field` at the accuracy of `settings` (its relative and
        /// absolute tolerance, and the order range whose level history the
        /// error estimates reach over).
        TimeStepper(TimeDependentField& field, const TimeSettings& settings);

        /// Record the current solution (the initial state, already prepared
        /// by the caller) as the first level, at time `t0`.
        void start(double t0);

        /// Advance one step, to time `t`, of BDF order `order` (1 or 2): the
        /// order the case's controller selected. The size of the step is the
        /// difference from the level the stepper is at.
        void step(double t, int order);

        /// The reference solver's error estimates `err_k`, `err_{k-1}` and
        /// `err_{k+1}` of the step just taken, with `k` the order it was taken
        /// at — the field's own, which the case combines over its fields (see
        /// `CaseScheduler::combine_errors`). Each is weighted per dof by
        /// `tolerance (absolute_factor + |u_i|)` and measured as the RMS over
        /// the dofs. That weight is the reference's `A + R|U_i|`: its absolute
        /// tolerance `A = R * absolute_factor` is what holds a dof whose value
        /// is near zero — round-off around a clamped face, a field that starts
        /// from nothing — to the accuracy of the field's own tolerance rather
        /// than to a relative accuracy on a value that carries nothing, which
        /// is what let the displacement's near-zero dofs set the step of the
        /// whole coupled case.
        ///
        /// The estimates are the reference solver's own, not the textbook
        /// local truncation error — see `StepError`, which defines them.
        StepError error_estimates() const;

        /// The weighted RMS norm of the time derivative the first level
        /// starts with, measured over the newest step: the `yp_norm` of the
        /// initial-step rule `DT yp_norm <= 1/2`.
        double derivative_norm() const;

        /// The solution at time `t` of the scheme's own polynomial through the
        /// levels it holds — what a transient result is reported with at an
        /// output time the steps stepped over.
        void interpolate(double t, la::Vector<double>& out) const;

        /// Write the newest level back into `out`, undoing `interpolate`.
        void restore(la::Vector<double>& out) const;

        /// The solution vector the stepper advances, which `interpolate` and
        /// `restore` write.
        la::Vector<double>& solution() { return *field_.solution()->x(); }

        /// Drop the step just taken, back to the level it started from: the
        /// driver rejected it, and the next attempt starts from there.
        void undo();

    private:
        /// The stored levels, most recent first.
        std::vector<const la::Vector<double>*> levels() const;

        /// The scheme's weights for a step of `steps`, taken at `order`.
        TimeWeights weights(int order, TimeSteps steps) const;

        /// The weighted RMS norm of `scale * values`, dof `i` weighted by
        /// `tolerance (absolute_factor + |u_i|)` with `u` the newest level.
        double weighted_norm(double scale, const la::Vector<double>& values) const;

        /// The reference solver's error estimate `err_order` at the order
        /// `order`: the weighted RMS norm of `order! d_(order+1)`, the scaled
        /// divided difference of the levels (see `StepError`). Zero when the
        /// history does not reach `order + 2` levels back.
        double error_norm(int order) const;

        TimeDependentField& field_;
        double tolerance_ = 0.0;
        double absolute_factor_ = 0.0;
        int max_order_ = 1;
        /// The levels the scheme and the error estimates read: the one just
        /// taken plus as many before it as the order above the one in use
        /// needs (its estimate reaches one level further than the scheme, and
        /// one further still than the estimate at the order below).
        std::size_t keep_levels_ = 0;
        /// Solutions of the levels, most recent first, and their times.
        std::vector<la::Vector<double>> history_;
        std::vector<double> times_;
        /// Order of the step just taken, which its error estimates use.
        int order_ = 1;
        /// Whether the history holds a step that `undo` can drop.
        bool pending_ = false;
    };

} // namespace hellofem::app
