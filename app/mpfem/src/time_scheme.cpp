// hellofem::app — time stepping schemes for transient solves
// SPDX-License-Identifier: MIT

#include "time_scheme.h"

#include <algorithm>
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

        /// Backward Euler: (M/dt + K) u^{n+1} = M u^n/dt + f^{n+1}.
        class Bdf1Scheme final : public TimeScheme {
        public:
            std::string_view name() const override { return "bdf1"; }
            int order() const override { return 1; }
            int levels() const override { return 2; }
            void weights(double dt, int, TimeWeights& w) const override
            {
                w.a = {1.0 / dt, -1.0 / dt};
                w.b = {1.0, 0.0};
                w.c_new = 1.0;
                w.c_old = 0.0;
            }
        };

        /// Crank-Nicolson (trapezoidal): the stiffness and the source are
        /// averaged between the two levels, so the scheme is second order.
        class CrankNicolsonScheme final : public TimeScheme {
        public:
            std::string_view name() const override { return "cn"; }
            int order() const override { return 2; }
            int levels() const override { return 2; }
            void weights(double dt, int, TimeWeights& w) const override
            {
                w.a = {1.0 / dt, -1.0 / dt};
                w.b = {0.5, 0.5};
                w.c_new = 0.5;
                w.c_old = 0.5;
            }
        };

        /// Second-order backward differentiation with a constant step. The
        /// first step has no u^{n-1}, so it falls back to backward Euler.
        class Bdf2Scheme final : public TimeScheme {
        public:
            std::string_view name() const override { return "bdf2"; }
            int order() const override { return 2; }
            int levels() const override { return 3; }
            void weights(double dt, int history, TimeWeights& w) const override
            {
                if (history < 2) {
                    Bdf1Scheme {}.weights(dt, history, w);
                    return;
                }
                w.a = {1.5 / dt, -2.0 / dt, 0.5 / dt};
                w.b = {1.0, 0.0, 0.0};
                w.c_new = 1.0;
                w.c_old = 0.0;
            }
        };

    } // namespace

    std::unique_ptr<const TimeScheme> make_time_scheme(std::string_view name)
    {
        const std::string key = lower(name);
        if (key == "bdf1" or key == "backward-euler" or key == "be")
            return std::make_unique<const Bdf1Scheme>();
        if (key == "cn" or key == "crank-nicolson" or key == "trapezoidal")
            return std::make_unique<const CrankNicolsonScheme>();
        if (key == "bdf2")
            return std::make_unique<const Bdf2Scheme>();
        throw std::runtime_error("unknown time stepping scheme '" + std::string(name)
            + "' (expected bdf1, cn or bdf2)");
    }

    std::vector<std::string> time_scheme_names()
    {
        return {"bdf1", "cn", "bdf2"};
    }

} // namespace hellofem::app
