// hellofem::app — time stepping driver of a time-dependent field
// SPDX-License-Identifier: MIT

#include "transient.h"

#include "solver.h"

#include <spdlog/spdlog.h>

#include <utility>

namespace hellofem::app {

    TimeStepper::TimeStepper(TimeDependentField& field, std::string_view scheme)
        : field_(field)
        , scheme_(find_time_scheme(scheme))
    {
    }

    void TimeStepper::start(double t0)
    {
        history_.push_back(la::Vector<double>(*field_.solution()->x()));
        source_.reset();
        t_ = t0;
    }

    void TimeStepper::step(double t)
    {
        const double dt = t - t_;
        const int previous = static_cast<int>(history_.size());
        const TimeWeights w = time_weights(scheme_, dt, previous);

        std::vector<const la::Vector<double>*> history;
        for (std::size_t k = 1; k < w.a.size() and k <= history_.size(); ++k)
            history.push_back(&history_[k - 1]);

        la::Vector<double> source(field_.space()->dofmap()->index_map,
            field_.space()->dofmap()->index_map_bs());
        TimeLevel level;
        level.weights = w;
        level.time = t;
        level.history = history;
        level.source_old = source_ ? &*source_ : nullptr;
        level.source_new = &source;
        const int iterations = solve_system(
            [&](la::MatrixCSR<double>& A, la::Vector<double>& b) {
                field_.refresh(t);
                field_.assemble_step(A, b, level);
            },
            *field_.solution()->x(), field_.pattern(), field_.nonlinear(),
            /*warm_start=*/true);

        spdlog::info("stepping '{}' to t = {} s (dt = {} s, {} iterations)",
            scheme_.name, t, dt, iterations);

        source_ = source;
        history_.insert(history_.begin(),
            la::Vector<double>(*field_.solution()->x()));
        const std::size_t keep
            = static_cast<std::size_t>(scheme_.levels - 1);
        if (history_.size() > keep)
            history_.erase(history_.begin() + static_cast<std::ptrdiff_t>(keep),
                history_.end());
        t_ = t;
    }

} // namespace hellofem::app
