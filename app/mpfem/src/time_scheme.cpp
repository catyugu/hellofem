// hellofem::app — the BDF time discretization and step control of a transient solve
// SPDX-License-Identifier: MIT

#include "time_scheme.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace hellofem::app {
    namespace {

        /// The Newton table of a divided difference: one row of level values
        /// per level, most recent first, rewritten in place by the passes of
        /// ascending order.
        using LevelTable = std::vector<std::vector<double>>;

        /// The newest `rows` levels as the starting rows of a table.
        LevelTable level_rows(
            std::span<const la::Vector<double>* const> levels, std::size_t rows)
        {
            LevelTable d(rows);
            for (std::size_t i = 0; i < rows; ++i)
                d[i].assign(levels[i]->array().begin(), levels[i]->array().end());
            return d;
        }

        /// The pass of order `m` of the table: afterwards `d[i]` holds the
        /// divided difference of order m over times[i] ... times[i + m], for
        /// every row the table holds an `i + m` for.
        void newton_pass(
            LevelTable& d, std::span<const double> times, std::size_t m)
        {
            for (std::size_t i = 0; i + m < d.size(); ++i) {
                const double span = times[i] - times[i + m];
                const double factor = span == 0.0 ? 0.0 : 1.0 / span;
                for (std::size_t j = 0; j < d[i].size(); ++j)
                    d[i][j] = (d[i][j] - d[i + 1][j]) * factor;
            }
        }

        // The step-size control of the reference's own implicit solver (IDA):
        // the factor a step that met the tolerance is grown by is the clamp of
        // the asymptotic estimate to [eta_min, eta_low] below one, one inside
        // the band (eta_min_fx, eta_max_fx), and eta_max above it.
        constexpr double eta_max_fx = 2.0;
        constexpr double eta_min_fx = 1.0;
        constexpr double eta_max = 2.0;
        constexpr double eta_low = 0.9;
        constexpr double eta_min = 0.5;
        /// The largest reduction of a failed step.
        constexpr double eta_min_ef = 0.25;

    } // namespace

    TimeWeights bdf_weights(int order, TimeSteps steps)
    {
        const double h = steps.dt;
        const double h_prev = steps.dt_previous;
        TimeWeights w;

        // Backward Euler. A second-order step with no step before it (the
        // first one of a run) is taken here as well: the quadratic it would
        // differentiate needs a level it does not have yet.
        if (order <= 1 or h_prev <= 0.0) {
            w.a = {1.0 / h, -1.0 / h};
            w.b = {1.0, 0.0};
            return w;
        }

        w.a = {(2.0 * h + h_prev) / (h * (h + h_prev)),
            -(h + h_prev) / (h * h_prev),
            h / (h_prev * (h + h_prev))};
        w.b = {1.0, 0.0, 0.0};
        return w;
    }

    TimeLevel steady_level(double t)
    {
        // The mass weight is stated as zero rather than left out: a physics
        // reads `weights.a[0]` to decide whether it has a mass operator at
        // all.
        TimeLevel level;
        level.weights.a = {0.0};
        level.weights.b = {1.0};
        level.time = t;
        return level;
    }

    BdfController::BdfController(const TimeSettings& settings, double span)
        : settings_(settings)
        , span_(span)
        , largest_(span * max_step_fraction)
        , smallest_(span * smallest_step_fraction)
    {
        if (settings_.min_order < 1 or settings_.min_order > settings_.max_order)
            throw std::runtime_error("BDF order range "
                + std::to_string(settings_.min_order) + ".."
                + std::to_string(settings_.max_order)
                + " is not a range of orders");
        if (settings_.max_order > 2)
            throw std::runtime_error("BDF order "
                + std::to_string(settings_.max_order)
                + " is above the highest the app implements (2)");
        if (settings_.steps == StepsMode::manual and settings_.manual_step <= 0.0)
            throw std::runtime_error(
                "a manual time step must be a positive step, not "
                + std::to_string(settings_.manual_step));
    }

    double BdfController::first_step(double derivative_norm) const
    {
        const double bounded = span_ * first_step_fraction;
        if (derivative_norm <= 0.0)
            return bounded;
        return std::min(bounded, 0.5 / derivative_norm);
    }

    void BdfController::start(double h)
    {
        h_ = std::clamp(h, smallest_, largest_);
        order_ = 1;
        order_used_ = 0;
        ns_ = 0;
        steps_ = 0;
        failures_ = 0;
        startup_ = true;
    }

    double BdfController::step_size() const
    {
        return settings_.steps == StepsMode::manual ? settings_.manual_step : h_;
    }

    int BdfController::order() const
    {
        return settings_.steps == StepsMode::manual ? settings_.max_order : order_;
    }

    bool BdfController::error_controlled() const
    {
        return settings_.steps != StepsMode::manual;
    }

    int BdfController::lowered_order(const StepError& error) const
    {
        // The reference compares the truncation error norms `terr_j =
        // (j + 1) err_j` and lowers the order when the one below is at least
        // twice as small, up to the order 2 — the top of the app's range. Its
        // rule for the orders above 2 reads one estimate further back than the
        // history keeps, and the orders above 2 are refused (see
        // `TimeSettings`).
        if (order_ == 1)
            return order_;
        return 2.0 * error.below <= 0.5 * 3.0 * error.at ? order_ - 1 : order_;
    }

    double BdfController::factor(double error, int order) const
    {
        const double estimate
            = std::pow(2.0 * error + 1e-4, -1.0 / (order + 1));
        if (estimate >= eta_max_fx)
            return std::min(estimate, eta_max);
        if (estimate <= eta_min_fx)
            return std::max(std::min(estimate, eta_low), eta_min);
        return 1.0;
    }

    void BdfController::accept(const StepError& error, double h)
    {
        ++steps_;
        failures_ = 0;

        // The steps taken at a constant step size and order, up to the order
        // of the step before this one plus two: the run the order selection
        // requires before it reconsiders. A driver that limits the step (a
        // mode's own output times) breaks the run, which is what the
        // selection's constant-step assumption needs.
        if (h != used_ or order_ != order_used_)
            ns_ = 0;
        ns_ = std::min(ns_ + 1, order_used_ + 2);

        const int raised = order_ - order_used_;
        order_used_ = order_;
        used_ = h;

        if (settings_.steps == StepsMode::manual)
            return; // the step and the order are the model's

        // The startup phase ends at the maximum order, at a failure (see
        // `reject`) or at a lowered order.
        if (lowered_order(error) == order_ - 1 or order_ == settings_.max_order)
            startup_ = false;

        if (startup_) {
            // Until the order reaches its maximum, every step is taken one
            // order higher and at twice the size. The first step has no
            // history behind it at all, so it is left as it is.
            if (steps_ > 1) {
                ++order_;
                h_ = std::min(2.0 * h, largest_);
            }
            return;
        }

        int action = 0; // -1 lower, 0 keep, +1 raise
        if (lowered_order(error) == order_ - 1)
            action = -1;
        else if (order_ == settings_.max_order)
            action = 0;
        else if (order_ + 1 >= ns_ or raised == 1)
            action = 0; // the step size has not settled, or the order just changed
        else if (3.0 * error.above < 0.5 * 2.0 * error.at)
            action = 1; // the order 1 raises when the estimate above is 1/3 of it

        double next = error.at;
        if (action < 0 and order_ > settings_.min_order) {
            --order_;
            next = error.below;
        }
        else if (action > 0) {
            ++order_;
            next = error.above;
        }

        h_ = std::min(factor(next, order_) * h, largest_);
    }

    void BdfController::reject(const StepError& error, double h)
    {
        // A failed step ends the startup phase, and its retry is smaller — by
        // the asymptotic factor at the first failure, by a quarter of the step
        // from the second on, and at the order 1 from the third.
        startup_ = false;
        ++failures_;
        const bool lower = lowered_order(error) == order_ - 1;
        if (failures_ >= 3)
            order_ = 1;
        else if (lower)
            order_ = order_ - 1;
        order_ = std::max(order_, settings_.min_order);
        order_used_ = order_;

        double eta = eta_min_ef;
        if (failures_ == 1) {
            const double estimate = lower ? error.below : error.at;
            eta = std::clamp(
                0.9 * std::pow(2.0 * estimate + 1e-4, -1.0 / (order_ + 1)),
                eta_min_ef, eta_low);
        }
        h_ = std::max(eta * h, smallest_);
    }

    la::Vector<double> divided_difference(int k,
        std::span<const la::Vector<double>* const> levels,
        std::span<const double> times)
    {
        // `levels` always holds the new level; the call sites keep a history.
        la::Vector<double> out(levels[0]->index_map(), levels[0]->bs());
        out.set(0);
        const std::size_t n = levels[0]->array().size();
        if (n == 0 or levels.size() != times.size() or k < 1
            or levels.size() < static_cast<std::size_t>(k) + 1)
            return out;

        // DD_k reads the newest k + 1 levels and no further ones.
        LevelTable d = level_rows(levels, static_cast<std::size_t>(k) + 1);
        for (std::size_t m = 1; m < d.size(); ++m)
            newton_pass(d, times, m);

        // Scale by the step into the newest level, so that the result is the
        // term of the Taylor expansion the order selection compares.
        const double dt = times[0] - times[1];
        const double scale = std::pow(dt, k);
        auto& values = out.array();
        for (std::size_t j = 0; j < n; ++j)
            values[j] = scale * d[0][j];
        return out;
    }

    la::Vector<double> interpolate_levels(
        std::span<const la::Vector<double>* const> levels,
        std::span<const double> times, double t)
    {
        la::Vector<double> out(levels[0]->index_map(), levels[0]->bs());
        out.set(0);
        const std::size_t n = levels[0]->array().size();
        if (n == 0 or levels.empty() or levels.size() != times.size())
            return out;

        // The Newton form, whose passes produce the divided differences one
        // order at a time: p(t) = u[t_0] + (t - t_0) u[t_0,t_1] + ...
        LevelTable d = level_rows(levels, levels.size());
        auto& values = out.array();
        std::vector<double> product(n, 1.0);
        for (std::size_t j = 0; j < n; ++j)
            values[j] = d[0][j];
        for (std::size_t m = 1; m < d.size(); ++m) {
            newton_pass(d, times, m);
            for (std::size_t j = 0; j < n; ++j) {
                product[j] *= t - times[m - 1];
                values[j] += product[j] * d[0][j];
            }
        }
        return out;
    }

} // namespace hellofem::app
