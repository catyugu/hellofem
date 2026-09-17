// hellofem::app — time stepping schemes for transient solves
// SPDX-License-Identifier: MIT
#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace hellofem::app {

    /// Weights of the time discretization of a first-order-in-time field
    /// equation `M u' + K u = f` at the new level `t_{n+1}`, with the step
    /// `dt`, written as
    ///     Σ_k a_k M u^{n+1-k} + Σ_k b_k K u^{n+1-k}
    ///         = c_new f(t_{n+1}) + c_old f(t_n).
    /// `a` and `b` are already divided by dt and index 0 is the new level;
    /// the schemes implemented here need the source at two levels only.
    struct TimeWeights {
        std::vector<double> a; // mass-operator weights
        std::vector<double> b; // stiffness-operator weights
        double c_new = 1.0;
        double c_old = 0.0;
        /// Fraction of the step at which Dirichlet data is evaluated,
        /// `t_n + theta*dt`: 1 for the BDF schemes (the new level),
        /// 1/2 for Crank-Nicolson (the midpoint, which keeps its second
        /// order with a time-dependent boundary value).
        double theta = 1.0;
    };

    /// A time stepping scheme: the temporal discretization of a
    /// first-order-in-time field equation.
    class TimeScheme {
    public:
        virtual ~TimeScheme() = default;

        /// Scheme name ("bdf1", "cn", "bdf2").
        virtual std::string_view name() const = 0;

        /// Order of accuracy in dt.
        virtual int order() const = 0;

        /// Number of solution levels the scheme uses (the new one included).
        virtual int levels() const = 0;

        /// Weights of the step from `t_n` to `t_n + dt`.
        /// @param[in] dt Step size.
        /// @param[in] history Number of previous solution levels available
        ///   (0 when only the initial state exists). A scheme that needs
        ///   more levels falls back to a lower-order step for the start-up.
        /// @param[out] w The weights.
        virtual void weights(double dt, int history, TimeWeights& w) const = 0;
    };

    /// Select a scheme by name (case-insensitive): "bdf1" (backward Euler),
    /// "cn" (Crank-Nicolson) or "bdf2" (second-order backward
    /// differentiation). Throws for an unknown name.
    std::unique_ptr<const TimeScheme> make_time_scheme(std::string_view name);

    /// Names of the available schemes.
    std::vector<std::string> time_scheme_names();

} // namespace hellofem::app
