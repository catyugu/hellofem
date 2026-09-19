// hellofem::app — time stepping driver of a time-dependent field
// SPDX-License-Identifier: MIT

#include "transient.h"

#include "solver.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <utility>

namespace hellofem::app {

    TimeStepper::TimeStepper(TimeDependentField& field, const TimeSettings& settings)
        : field_(field)
        , tolerance_(settings.tolerance)
        , absolute_factor_(settings.absolute_factor)
        , max_order_(settings.max_order)
        , keep_levels_(static_cast<std::size_t>(settings.max_order) + 2)
    {
    }

    void TimeStepper::start(double t0)
    {
        history_.clear();
        times_.clear();
        sources_.clear();
        history_.push_back(la::Vector<double>(*field_.solution()->x()));
        times_.push_back(t0);
    }

    std::vector<const la::Vector<double>*> TimeStepper::levels() const
    {
        std::vector<const la::Vector<double>*> out;
        out.reserve(history_.size());
        for (const la::Vector<double>& level : history_)
            out.push_back(&level);
        return out;
    }

    TimeWeights TimeStepper::weights(int order, TimeSteps steps) const
    {
        // The order is the case's, up to the one the settings reach; a step
        // with no history behind it (or no step at all) is first order
        // whatever the request.
        return bdf_weights(std::min(order, max_order_), steps);
    }

    void TimeStepper::step(double t, int order)
    {
        const TimeSteps steps {t - times_.front(),
            times_.size() > 1 ? times_[0] - times_[1] : 0.0};
        const TimeWeights w = weights(order, steps);

        std::vector<const la::Vector<double>*> history;
        for (std::size_t k = 1; k < w.a.size() and k <= history_.size(); ++k)
            history.push_back(&history_[k - 1]);

        la::Vector<double> source(field_.space()->dofmap()->index_map,
            field_.space()->dofmap()->index_map_bs());
        TimeLevel level;
        level.weights = w;
        level.time = t;
        level.history = history;
        level.source_old = sources_.empty() ? nullptr : &sources_.front();
        level.source_new = &source;
        const int iterations = solve_system(
            [&](la::MatrixCSR<double>& A, la::Vector<double>& b) {
                field_.refresh(t);
                field_.assemble_step(A, b, level);
            },
            *field_.solution()->x(), field_.pattern(), field_.linear_solver(),
            field_.linear_settings());

        spdlog::debug("stepping to t = {} s (dt = {} s, order {}, {} iterations)",
            t, steps.dt, static_cast<int>(w.a.size()) - 1, iterations);

        history_.insert(history_.begin(),
            la::Vector<double>(*field_.solution()->x()));
        times_.insert(times_.begin(), t);
        sources_.insert(sources_.begin(), std::move(source));
        if (history_.size() > keep_levels_) {
            history_.erase(
                history_.begin() + static_cast<std::ptrdiff_t>(keep_levels_),
                history_.end());
            times_.erase(times_.begin() + static_cast<std::ptrdiff_t>(keep_levels_),
                times_.end());
            // One load fewer than levels: the first level has none.
            if (sources_.size() >= keep_levels_)
                sources_.erase(sources_.begin()
                        + static_cast<std::ptrdiff_t>(keep_levels_ - 1),
                    sources_.end());
        }
        order_ = static_cast<int>(w.a.size()) - 1;
        pending_ = true;
    }

    void TimeStepper::undo()
    {
        if (not pending_ or history_.size() < 2)
            return;
        *field_.solution()->x() = history_[1];
        history_.erase(history_.begin());
        times_.erase(times_.begin());
        if (not sources_.empty())
            sources_.erase(sources_.begin());
        pending_ = false;
    }

    double TimeStepper::weighted_norm(
        double scale, const la::Vector<double>& values) const
    {
        const auto& u = field_.solution()->x()->array();
        const auto& v = values.array();
        const std::size_t n = u.size();
        double sum = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const double weight = tolerance_ * (absolute_factor_ + std::abs(u[i]));
            const double ratio = weight > 0.0 ? scale * v[i] / weight : 0.0;
            sum += ratio * ratio;
        }
        return n > 0 ? std::sqrt(sum / static_cast<double>(n)) : 0.0;
    }

    double TimeStepper::error_norm(int order) const
    {
        // `err_order = order! |d_(order+1)|` reads order + 2 levels. A shorter
        // history — the first steps of a run, whose sizes the startup phase
        // sets — has no estimate at this order, and the step stands on the
        // tolerance it was proposed with.
        if (order < 1 or order > max_order_
            or times_.size() < static_cast<std::size_t>(order) + 2)
            return 0.0;
        double factorial = 1.0;
        for (int j = 2; j <= order; ++j)
            factorial *= static_cast<double>(j);
        return factorial
            * weighted_norm(1.0, divided_difference(order + 1, levels(), times_));
    }

    StepError TimeStepper::error_estimates() const
    {
        StepError out;
        out.at = error_norm(order_);
        if (order_ > 1)
            out.below = error_norm(order_ - 1);
        if (order_ < max_order_)
            out.above = error_norm(order_ + 1);
        return out;
    }

    double TimeStepper::derivative_norm() const
    {
        if (times_.size() < 2)
            return 0.0;
        const double dt = times_[0] - times_[1];
        if (dt <= 0.0)
            return 0.0;
        la::Vector<double> derivative(history_[0].index_map(), history_[0].bs());
        auto& d = derivative.array();
        const auto& u_new = history_[0].array();
        const auto& u_old = history_[1].array();
        for (std::size_t i = 0; i < d.size(); ++i)
            d[i] = (u_new[i] - u_old[i]) / dt;
        return weighted_norm(1.0, derivative);
    }

    void TimeStepper::interpolate(double t, la::Vector<double>& out) const
    {
        const std::vector<const la::Vector<double>*> level = levels();
        const std::size_t n = std::min(
            static_cast<std::size_t>(order_) + 1, level.size());
        if (n == 0 or times_.size() < n) {
            out.set(0.0);
            return;
        }
        out = interpolate_levels(
            std::span<const la::Vector<double>* const>(level.data(), n),
            std::span<const double>(times_.data(), n), t);
    }

    void TimeStepper::restore(la::Vector<double>& out) const
    {
        if (not history_.empty())
            out = history_.front();
    }

} // namespace hellofem::app
