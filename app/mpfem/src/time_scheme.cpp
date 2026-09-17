// hellofem::app — time stepping schemes for transient solves
// SPDX-License-Identifier: MIT

#include "time_scheme.h"

#include <algorithm>
#include <array>
#include <cctype>
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

        // The schemes: their weights, and the scheme a start-up step falls
        // back on while the history is shorter than they need.
        const std::array<TimeScheme, 3> schemes {{
            // Backward Euler: (M/dt + K) u^{n+1} = M u^n / dt + f^{n+1}.
            {"bdf1", 1, 2, {1.0, -1.0}, {1.0, 0.0}, 1.0, 0.0, {}},
            // Crank-Nicolson: the stiffness and the load are the average of
            // the two levels, which is what makes it second order.
            {"cn", 2, 2, {1.0, -1.0}, {0.5, 0.5}, 0.5, 0.5, {}},
            // BDF2, at a constant step.
            {"bdf2", 2, 3, {1.5, -2.0, 0.5}, {1.0, 0.0, 0.0}, 1.0, 0.0, "bdf1"},
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

    TimeWeights time_weights(const TimeScheme& scheme, double dt, int previous)
    {
        if (previous < scheme.levels - 1 and not scheme.startup.empty())
            return time_weights(find_time_scheme(scheme.startup), dt, previous);
        TimeWeights w;
        w.a.resize(scheme.a.size());
        w.b.resize(scheme.b.size());
        for (std::size_t k = 0; k < scheme.a.size(); ++k) {
            w.a[k] = scheme.a[k] / dt;
            w.b[k] = scheme.b[k];
        }
        w.c_new = scheme.c_new;
        w.c_old = scheme.c_old;
        return w;
    }

    std::string time_scheme_names()
    {
        std::string names;
        for (const TimeScheme& scheme : schemes)
            names += (names.empty() ? "" : ", ") + std::string(scheme.name);
        return names;
    }

} // namespace hellofem::app
