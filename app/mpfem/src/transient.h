// hellofem::app — time stepping driver of a time-dependent field
// SPDX-License-Identifier: MIT
#pragma once

#include "field.h"
#include "time_scheme.h"

#include <cstddef>
#include <optional>
#include <string_view>
#include <vector>

namespace hellofem::app {

    /// Advances a time-dependent field with a time stepping scheme.
    ///
    /// Owns the solution history and the source load of the previous level,
    /// which the scheme needs, and refreshes the field at every new level.
    /// The solution of the previous level is the initial guess of the next
    /// one — for the nonlinear material update and for the Krylov solve.
    class TimeStepper {
    public:
        /// Advance `field` in time with the scheme named `scheme` (see
        /// `time_schemes`).
        TimeStepper(TimeDependentField& field, std::string_view scheme);

        /// Record the current solution (the initial state, already
        /// prepared by the caller) as the first level, at time `t0`.
        void start(double t0);

        /// Advance one step, to the output time `t` (the step size is the
        /// difference from the previous level).
        void step(double t);

        const TimeScheme& scheme() const { return scheme_; }

    private:
        TimeDependentField& field_;
        const TimeScheme& scheme_;
        std::vector<la::Vector<double>> history_;
        std::optional<la::Vector<double>> source_;
        double t_ = 0.0;
    };

} // namespace hellofem::app
