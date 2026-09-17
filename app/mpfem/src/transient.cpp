// hellofem::app — time stepping driver of a heat-transfer solve
// SPDX-License-Identifier: MIT

#include "transient.h"

#include "solver.h"

#include <spdlog/spdlog.h>

#include <utility>

namespace hellofem::app {

    HeatTimeStepper::HeatTimeStepper(std::shared_ptr<HeatTransferSolver> solver,
        std::unique_ptr<const TimeScheme> scheme)
        : solver_(std::move(solver))
        , scheme_(std::move(scheme))
    {
    }

    void HeatTimeStepper::start(double t0)
    {
        history_.push_back(la::Vector<double>(*solver_->solution()->x()));
        source_.reset();
        t_ = t0;
    }

    void HeatTimeStepper::step(double t)
    {
        const double dt = t - t_;
        TimeWeights w;
        scheme_->weights(dt, static_cast<int>(history_.size()), w);

        std::vector<const la::Vector<double>*> history;
        for (std::size_t k = 1; k < w.a.size() and k <= history_.size(); ++k)
            history.push_back(&history_[k - 1]);

        la::Vector<double> source(solver_->space()->dofmap()->index_map,
            solver_->space()->dofmap()->index_map_bs());
        TimeLevel level;
        level.weights = w;
        level.time = t;
        level.history = history;
        level.source_old = source_ ? &*source_ : nullptr;
        level.source_new = &source;
        const int iterations = solve_system(
            [&](la::MatrixCSR<double>& A, la::Vector<double>& b) {
                solver_->refresh(t);
                solver_->assemble_step(A, b, level);
            },
            *solver_->solution()->x(), solver_->pattern(), solver_->nonlinear(),
            /*warm_start=*/true);

        spdlog::info("stepping '{}' to t = {} s (dt = {} s, {} iterations)",
            scheme_->name(), t, dt, iterations);

        source_ = source;
        history_.insert(history_.begin(),
            la::Vector<double>(*solver_->solution()->x()));
        const std::size_t keep
            = static_cast<std::size_t>(scheme_->levels() - 1);
        if (history_.size() > keep)
            history_.erase(history_.begin() + static_cast<std::ptrdiff_t>(keep),
                history_.end());
        t_ = t;
    }

} // namespace hellofem::app
