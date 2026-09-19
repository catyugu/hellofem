// hellofem::app — time stepping schemes and step-size control of a transient solve
// SPDX-License-Identifier: MIT

#include "time_scheme.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <stdexcept>

namespace hellofem::app {
    namespace {

        std::string lower(std::string_view text)
        {
            std::string out(text);
            std::ranges::transform(out, out.begin(), [](unsigned char c) {
                return static_cast<char>(std::tolower(c));
            });
            return out;
        }

        // The schemes: the family and the order of each. How the steps are
        // placed is the step control, not the scheme (see `TimeSettings`).
        const std::array<TimeScheme, 3> schemes {{
            {"bdf1", TimeFamily::bdf, 1},
            {"cn", TimeFamily::crank_nicolson, 2},
            {"bdf2", TimeFamily::bdf, 2},
        }};

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

    } // namespace

    std::span<const TimeScheme> time_schemes()
    {
        return schemes;
    }

    const TimeScheme& find_time_scheme(std::string_view name)
    {
        const std::string key = lower(name);
        for (const TimeScheme& scheme : schemes)
            if (scheme.name == key)
                return scheme;
        std::string names;
        for (const TimeScheme& scheme : schemes)
            names += (names.empty() ? "" : ", ") + std::string(scheme.name);
        throw std::runtime_error("unknown time stepping scheme '" + std::string(name)
            + "'; the available ones are " + names);
    }

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

    TimeWeights cn_weights(double dt)
    {
        TimeWeights w;
        w.a = {1.0 / dt, -1.0 / dt};
        w.b = {0.5, 0.5};
        w.c_new = 0.5;
        w.c_old = 0.5;
        return w;
    }

    double error_coefficient(TimeFamily family, int order)
    {
        if (family == TimeFamily::crank_nicolson)
            return 0.5; // 3! * 1/12
        return order <= 1 ? 1.0 : 4.0 / 3.0; // 2! * 1/2, 3! * 2/9
    }

    double step_factor(double error, int order)
    {
        if (error <= 0.0)
            return 2.0;
        // The elementary controller, applied to every step: the size the
        // asymptotic dependence of the local error on the step allows,
        // h_new = 0.9 error^(-1/(order+1)) h, with a safety factor and the
        // clamp [0.1, 2]. A step that met the tolerance therefore grows while
        // its estimate sits below 0.9^(order+1) of it (0.81 at order 1, 0.73 at
        // order 2) instead of being held until it is far inside.
        const double factor = 0.9 * std::pow(error, -1.0 / (order + 1));
        return std::max(0.1, std::min(2.0, factor));
    }

    int next_order(int order, int max_order, std::span<const double> derivative_scale)
    {
        // The norms are `|dt^k DD_k u|` at index k - 1, so the two the choice
        // reads are the ones around the current order.
        const auto norm = [&](int k) {
            return k >= 1 and k <= static_cast<int>(derivative_scale.size())
                ? derivative_scale[k - 1]
                : 0.0;
        };

        if (order < max_order and norm(order + 1) < norm(order))
            return order + 1;
        if (order > 1 and norm(order) > norm(order - 1))
            return order - 1;
        return order;
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

    std::vector<double> derivative_scale(
        std::span<const la::Vector<double>* const> levels,
        std::span<const double> times)
    {
        // One entry per order the level set supports: a set too short for an
        // order has no entry for it rather than a zero (see `next_order`).
        if (levels.size() != times.size() or levels.size() < 2)
            return {};

        const std::size_t n = levels[0]->array().size();
        std::vector<double> scale(levels.size() - 1, 0.0);
        LevelTable d = level_rows(levels, levels.size());
        const double dt = times[0] - times[1];
        for (std::size_t m = 1; m < levels.size(); ++m) {
            newton_pass(d, times, m);
            const double weight = std::pow(dt, static_cast<double>(m));
            for (std::size_t j = 0; j < n; ++j)
                scale[m - 1]
                    = std::max(scale[m - 1], std::abs(weight * d[0][j]));
        }
        return scale;
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
