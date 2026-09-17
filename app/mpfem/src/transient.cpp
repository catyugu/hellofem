// hellofem::app — time stepping driver of a time-dependent field
// SPDX-License-Identifier: MIT

#include "transient.h"

#include "solver.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <utility>

namespace hellofem::app {
    namespace {

        /// The leading coefficient of the local truncation error of a BDF step
        /// of order `order`, against the divided differences the estimate
        /// reads: that error is `C_q dt^(q+1) u^(q+1)` with C_1 = 1/2 and
        /// C_2 = 2/9, and `u^(q+1) = (q+1)! DD_{q+1}`, so the estimate is
        /// `c_q dt^(q+1) DD_(q+1)` with c_1 = 2! / 2 = 1 and
        /// c_2 = 3! * 2/9 = 4/3. The estimate is the leading term only, and
        /// it is what the step size is controlled by.
        double error_coefficient(int order)
        {
            return order <= 1 ? 1.0 : 4.0 / 3.0;
        }

        /// The absolute part of the error weight, as a fraction of the
        /// field's own scale: the weight is
        /// `tolerance |u| + tolerance * weight_floor * magnitude`.
        constexpr double weight_floor = 1e-3;

    } // namespace

    TimeStepper::TimeStepper(TimeDependentField& field, const TimeSettings& settings)
        : field_(field)
        , scheme_(find_time_scheme(settings.scheme))
        , tolerance_(settings.tolerance)
        , adaptive_(settings.adaptive)
    {
        // The estimate is the leading term of the BDF truncation error; the
        // trapezoidal scheme has no such estimate installed.
        if (adaptive_ and scheme_.family != TimeFamily::bdf)
            throw std::runtime_error("adaptive step control needs a BDF scheme: '"
                + std::string(scheme_.name) + "' has no truncation error estimate");
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
        // The order is the driver's only where the step control selects it;
        // a step with no history behind it (or no step at all) is first order
        // whatever the request.
        const int taken = adaptive_ ? std::min(order, scheme_.order)
                                    : scheme_.order;
        return bdf_weights(taken, steps);
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
            *field_.solution()->x(), field_.pattern(), field_.nonlinear(),
            /*warm_start=*/true);

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
        const double floor = tolerance_ * weight_floor * magnitude_;

        const double coefficient = error_coefficient(order_);
        double sum = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const double weight = tolerance_ * std::abs(u[i]) + floor;
            const double estimate = coefficient * dd.array()[i];
            const double ratio = weight > 0.0 ? estimate / weight : 0.0;
            sum += ratio * ratio;
        }
        return n > 0 ? std::sqrt(sum / static_cast<double>(n)) : 0.0;
    }

    std::vector<double> TimeStepper::scaled_derivatives() const
    {
        return derivative_scale(levels(), times_);
    }

} // namespace hellofem::app
