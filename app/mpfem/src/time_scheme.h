// hellofem::app — time stepping schemes for transient solves
// SPDX-License-Identifier: MIT
#pragma once

#include <span>
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
    };

    /// A time stepping scheme, described by its weights alone: a scheme is a
    /// row of the table returned by `time_schemes()`.
    struct TimeScheme {
        /// Name the `--scheme` option accepts.
        std::string_view name;
        /// Order of accuracy in dt.
        int order = 1;
        /// Solution levels the scheme uses, the new one included.
        int levels = 2;
        /// Mass weights, 1/dt not applied yet; index 0 is the new level.
        std::vector<double> a;
        /// Stiffness weights.
        std::vector<double> b;
        /// Load coefficients of the new and of the previous level.
        double c_new = 1.0;
        double c_old = 0.0;
        /// Scheme to fall back on while fewer than `levels` solutions are
        /// available (BDF2 starts on backward Euler); empty when the scheme
        /// applies from the first step.
        std::string_view startup;
    };

    /// The available schemes.
    std::span<const TimeScheme> time_schemes();

    /// The scheme named `name`, case-insensitive. Throws for an unknown name.
    const TimeScheme& find_time_scheme(std::string_view name);

    /// Weights of the step from `t_n` to `t_n + dt`, with `previous` past
    /// solutions available (fewer than the scheme needs selects its start-up
    /// fallback).
    TimeWeights time_weights(const TimeScheme& scheme, double dt, int previous);

    /// Comma-separated names of the available schemes.
    std::string time_scheme_names();

} // namespace hellofem::app
