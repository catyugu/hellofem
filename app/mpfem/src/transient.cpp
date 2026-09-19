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
        , scheme_(find_time_scheme(settings.scheme))
        , tolerance_(settings.tolerance)
    {
    }

    void TimeStepper::start(double t0)
    {
        history_.clear();
        times_.clear();
        sources_.clear();
        history_.push_back(la::Vector<double>(*field_.solution()->x()));
        times_.push_back(t0);
        magnitude_ = 0.0;
        track_magnitude();
    }

    void TimeStepper::track_magnitude()
    {
        for (const double value : field_.solution()->x()->array())
            magnitude_ = std::max(magnitude_, std::abs(value));
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
        if (scheme_.family == TimeFamily::crank_nicolson)
            return cn_weights(steps.dt);
        // The order is the driver's, up to the one the scheme integrates at;
        // a step with no history behind it (or no step at all) is first order
        // whatever the request.
        return bdf_weights(std::min(order, scheme_.order), steps);
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

        spdlog::debug("stepping '{}' to t = {} s (dt = {} s, order {}, {} iterations)",
            scheme_.name, t, steps.dt, static_cast<int>(w.a.size()) - 1,
            iterations);

        track_magnitude();
        history_.insert(history_.begin(),
            la::Vector<double>(*field_.solution()->x()));
        times_.insert(times_.begin(), t);
        sources_.insert(sources_.begin(), std::move(source));
        if (history_.size() > keep_levels) {
            history_.erase(history_.begin() + static_cast<std::ptrdiff_t>(keep_levels),
                history_.end());
            times_.erase(times_.begin() + static_cast<std::ptrdiff_t>(keep_levels),
                times_.end());
            // One load fewer than levels: the first level has none.
            if (sources_.size() >= keep_levels)
                sources_.erase(
                    sources_.begin() + static_cast<std::ptrdiff_t>(keep_levels - 1),
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

    double TimeStepper::error() const
    {
        if (times_.size() < static_cast<std::size_t>(order_) + 2
            or history_.empty())
            return 0.0; // too short a history to estimate: the step stands

        const std::vector<const la::Vector<double>*> level = levels();
        const la::Vector<double> dd
            = divided_difference(order_ + 1, level, times_);
        const auto& u = field_.solution()->x()->array();
        const std::size_t n = u.size();

        const double coefficient = error_coefficient(scheme_.family, order_);
        // The weight of a dof is the field's own scale, or the dof's value
        // where that is larger: `tolerance max(|u_i|, scale)`, COMSOL's
        // `W_ij = max(|U_ij|, S_j)` (see its "Scales for dependent variables").
        // The scale is what keeps a field with a wide range of values — a
        // displacement that runs from 1e-11 near a clamped face to its own
        // magnitude — from letting its near-zero dofs, which carry nothing but
        // round-off, set the step of the whole case.
        const double scale = magnitude_;
        double sum = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const double weight = tolerance_ * std::max(std::abs(u[i]), scale);
            const double ratio
                = weight > 0.0 ? coefficient * dd.array()[i] / weight : 0.0;
            sum += ratio * ratio;
        }
        return n > 0 ? std::sqrt(sum / static_cast<double>(n)) : 0.0;
    }

    std::vector<double> TimeStepper::scaled_derivatives() const
    {
        return derivative_scale(levels(), times_);
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
