// hellofem::app — time stepping driver of a time-dependent field
// SPDX-License-Identifier: MIT
#pragma once

#include "field.h"
#include "time_scheme.h"

#include <cstddef>
#include <vector>

namespace hellofem::app {

    /// Advances a time-dependent field with a time stepping scheme.
    ///
    /// Owns the solution history and the source loads of the last levels,
    /// which the scheme and the error estimate read, and refreshes the field
    /// at every new level. The solution of the previous level is the initial
    /// guess of the next one — for the nonlinear material update and for the
    /// Krylov solve.
    ///
    /// A step is kept until the driver either accepts it (the next step is
    /// taken from it) or drops it with `undo`; the driver judges it with
    /// `error`, and selects the order of the next one with
    /// `scaled_derivatives`.
    class TimeStepper {
    public:
        /// Advance `field` in time with the scheme and the tolerance of
        /// `settings`.
        TimeStepper(TimeDependentField& field, const TimeSettings& settings);

        /// Record the current solution (the initial state, already prepared
        /// by the caller) as the first level, at time `t0`.
        void start(double t0);

        /// Advance one step, to time `t`, of BDF order `order` (1 or 2). The
        /// size of the step is the difference from the level the stepper is
        /// at; a stepper without adaptive control takes every step at the
        /// order of its scheme, and a scheme of another family than BDF
        /// ignores the order and its own weights are used.
        void step(double t, int order);

        /// The local truncation error of the step just taken, weighted per
        /// dof by `tolerance |u| + tolerance * 1e-3 * max|u|` and measured as
        /// the RMS over the dofs: the step meets the tolerance while this is
        /// at most one. The weight is relative wherever the solution is of
        /// the level's magnitude, and the small absolute part keeps a dof the
        /// solution has barely reached from demanding an accuracy no step
        /// could deliver (a field that starts at zero).
        double error() const;

        /// The scaled derivative norms `|dt^k DD_k u|` of the level just
        /// taken, for the order selection (see `next_order`).
        std::vector<double> scaled_derivatives() const;

        /// Drop the step just taken, back to the level it started from: the
        /// driver rejected it, and the next attempt starts from there.
        void undo();

        const TimeScheme& scheme() const { return scheme_; }

    private:
        /// The levels the scheme and the error estimate read: the one just
        /// taken plus the three before it, which is what a second-order
        /// estimate needs (it reaches one level further than the scheme).
        static constexpr std::size_t keep_levels = 4;

        /// The stored levels, most recent first.
        std::vector<const la::Vector<double>*> levels() const;

        /// The scheme's weights for a step of `steps`, taken at `order`.
        TimeWeights weights(int order, TimeSteps steps) const;

        /// Take the magnitude of the current solution as the field's scale,
        /// when it is the largest one yet.
        void track_magnitude();

        TimeDependentField& field_;
        const TimeScheme& scheme_;
        double tolerance_ = 1e-3;
        /// Whether the step size and the order of every step are the
        /// driver's, i.e. whether it may read `error` and
        /// `scaled_derivatives` for them.
        bool adaptive_ = false;
        /// Solutions of the levels, most recent first, and their times.
        std::vector<la::Vector<double>> history_;
        std::vector<double> times_;
        /// Source load of every level but the first, in the same order as
        /// `times_` without its first entry: the load of the level a step
        /// starts from is what its scheme reads as the old one.
        std::vector<la::Vector<double>> sources_;
        /// Order of the step just taken, which its error estimate uses.
        int order_ = 1;
        /// The largest magnitude the solution has reached, over every level of
        /// the run: the field's own scale, which the absolute part of the
        /// error weight is set from. It only grows — an undone step leaves it
        /// where it is, which holds the field to the accuracy of a magnitude
        /// it has really had — so a solution that starts at zero, or decays
        /// towards it, is never measured against a relative criterion on a
        /// value that carries nothing left.
        double magnitude_ = 0.0;
        /// Whether the history holds a step that `undo` can drop.
        bool pending_ = false;
    };

} // namespace hellofem::app
