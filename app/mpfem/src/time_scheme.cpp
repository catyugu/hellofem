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
        throw std::runtime_error("unknown time stepping scheme '" + std::string(name)
            + "'; the available ones are " + time_scheme_names());
    }

    std::string time_scheme_names()
    {
        std::string names;
        for (const TimeScheme& scheme : schemes)
            names += (names.empty() ? "" : ", ") + std::string(scheme.name);
        return names;
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

    double step_factor(double error, int order)
    {
        if (error > 1.0) {
            const double factor = 0.9 * std::pow(error, -1.0 / (order + 1));
            return std::max(0.1, std::min(1.0, factor));
        }
        // A step that met the tolerance, with room: the deadbeat region of
        // the controller is everything down to a sixteenth of it.
        return error <= 1.0 / 16.0 ? 2.0 : 1.0;
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

        // Newton divided differences, ascending in the level order: after the
        // pass of order m, `d[i]` holds DD_m over times[i] ... times[i + m].
        std::array<std::vector<double>, 4> d;
        for (std::size_t i = 0; i < levels.size(); ++i)
            d[i].assign(levels[i]->array().begin(), levels[i]->array().end());

        for (int m = 1; m <= k; ++m)
            for (std::size_t i = 0; i + static_cast<std::size_t>(m) < levels.size();
                ++i) {
                const double span = times[i] - times[i + static_cast<std::size_t>(m)];
                const double factor = span == 0.0 ? 0.0 : 1.0 / span;
                for (std::size_t j = 0; j < n; ++j)
                    d[i][j] = (d[i][j] - d[i + 1][j]) * factor;
            }

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
        std::vector<double> scale(3, 0.0);
        for (int k = 1; k <= 3; ++k) {
            const la::Vector<double> dd = divided_difference(k, levels, times);
            for (double value : dd.array())
                scale[static_cast<std::size_t>(k - 1)]
                    = std::max(scale[static_cast<std::size_t>(k - 1)],
                        std::abs(value));
        }
        return scale;
    }

} // namespace hellofem::app
