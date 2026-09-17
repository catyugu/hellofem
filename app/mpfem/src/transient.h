// hellofem::app — time stepping driver of a heat-transfer solve
// SPDX-License-Identifier: MIT
#pragma once

#include "physics.h"
#include "time_scheme.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

namespace hellofem::app {

    /// Advances a heat-transfer solve in time with a time stepping scheme.
    ///
    /// Owns the solution history and the source load of the previous level,
    /// which the scheme needs, and refreshes the solver at every new level.
    /// The solution of the previous level is the initial guess of the next
    /// one — for the nonlinear material update and for the Krylov solve.
    class HeatTimeStepper {
    public:
        HeatTimeStepper(std::shared_ptr<HeatTransferSolver> solver,
            std::unique_ptr<const TimeScheme> scheme);

        /// Record the current solution (the initial state, already
        /// prepared by the caller) as the first level, at time `t0`.
        void start(double t0);

        /// Advance one step, to the output time `t` (the step size is the
        /// difference from the previous level).
        void step(double t);

        const TimeScheme& scheme() const { return *scheme_; }

    private:
        std::shared_ptr<HeatTransferSolver> solver_;
        std::unique_ptr<const TimeScheme> scheme_;
        std::vector<la::Vector<double>> history_;
        std::optional<la::Vector<double>> source_;
        double t_ = 0.0;
    };

} // namespace hellofem::app
