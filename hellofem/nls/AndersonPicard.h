// hellofem::nls — Anderson-accelerated Picard (fixed-point) iteration
// SPDX-License-Identifier: MIT

#pragma once

#include "la/KrylovSolver.h"
#include "la/MatrixCSR.h"
#include "la/Vector.h"

#include <Eigen/Dense>

#include <cmath>
#include <cstdint>
#include <deque>
#include <functional>
#include <optional>
#include <stdexcept>
#include <utility>

#include <spdlog/spdlog.h>

namespace hellofem::nls {

    /// Configuration of the Anderson-accelerated Picard iteration.
    ///
    /// Mirrors the MetaHotspot nonlinear solver settings. During the
    /// first `warmup_iters` iterations (and whenever `max_growth` trips)
    /// the iteration falls back to a damped Picard step
    /// `x += dampening * (G(x) - x)`.
    struct AndersonConfig {
        /// Number of past (x, G) pairs kept for the Anderson mixing.
        int depth = 5;

        /// Number of initial plain Picard iterations before mixing.
        int warmup_iters = 2;

        /// Anderson damping (1.0 = full AA, 0.0 = plain Picard). Also the
        /// under-relaxation used for the damped-Picard fallback.
        double dampening = 0.8;

        /// Divergence guard: reject the mixed step if its infinity norm
        /// exceeds `max_growth` times the plain step's.
        double max_growth = 1.5;

        /// Reset the mixing history when the divergence guard trips.
        bool reset_on_growth = true;

        /// Relative tolerance on the residual and on the update.
        double relative_tolerance = 1e-6;

        /// Absolute tolerance on the residual and on the update.
        double absolute_tolerance = 1e-12;

        /// Maximum number of Picard iterations.
        int max_iterations = 200;

        /// Log each iteration.
        bool report = true;

        /// Inner linear solver settings for the frozen system `A G = b`.
        std::string linear_solver_type = "cg";
        std::string preconditioner_type = "jacobi";
        double krylov_rtol = 1e-12;
        double krylov_atol = 1e-14;
        int krylov_max_iter = 1000;

        /// Called after each inner linear solve, before the mixing step
        /// (e.g. to apply Dirichlet lifting or update auxiliary state).
        std::function<void()> post_linear_solve = [] {};
    };

    /// Result of an Anderson-accelerated Picard solve.
    template <std::floating_point T>
    struct AndersonResult {
        /// Whether the iteration converged.
        bool converged = false;

        /// Number of Picard iterations used.
        int iterations = 0;

        /// Number of inner linear-solver iterations accumulated.
        int krylov_iterations = 0;
    };

    /// Anderson mixing for a fixed-point iteration `x <- G(x)`.
    ///
    /// Given the current iterate `x_k` and its fixed-point image `G_k`,
    /// solves the least-squares problem `F alpha ~= f_k` for the mixing
    /// coefficients over the last `depth` iterates and proposes the mixed
    /// iterate. History index 0 is the most recent pair.
    ///
    /// @tparam T Scalar type.
    template <std::floating_point T>
    class AndersonMixer {
    public:
        /// Create a mixer.
        /// @param[in] depth History depth.
        /// @param[in] warmup_iters Plain-Picard iterations before mixing.
        /// @param[in] dampening Mixing damping.
        /// @param[in] max_growth Divergence-guard ratio (> 0 to enable).
        /// @param[in] reset_on_growth Clear history when the guard trips.
        AndersonMixer(int depth, int warmup_iters, double dampening,
            double max_growth, bool reset_on_growth)
            : _depth(depth), _warmup_iters(warmup_iters), _dampening(dampening),
              _max_growth(max_growth), _reset_on_growth(reset_on_growth)
        {
        }

        /// Record the pair `(x_k, G_k)` and advance the iteration counter.
        void push(const la::Vector<T>& x_k, const la::Vector<T>& G_k)
        {
            _x_hist.push_front(x_k);
            _G_hist.push_front(G_k);
            if (static_cast<int>(_x_hist.size()) > _depth) {
                _x_hist.pop_back();
                _G_hist.pop_back();
            }
            ++_iter_count;
        }

        /// Propose the next iterate, or `std::nullopt` to fall back to a
        /// plain (damped) Picard step.
        /// @param[in] x_k Current iterate.
        /// @param[in] G_k Fixed-point image `G(x_k)`.
        /// @return The mixed next iterate, or `std::nullopt`.
        std::optional<la::Vector<T>> step(const la::Vector<T>& x_k,
            const la::Vector<T>& G_k) const
        {
            // Disabled, still warming up, or no history: plain Picard.
            if (_depth == 0 or _iter_count < _warmup_iters
                or _G_hist.empty()) {
                return std::nullopt;
            }

            const int n = static_cast<int>(x_k.array().size());
            const int m_k = std::min(
                static_cast<int>(_G_hist.size()), _depth);

            // Residual of the fixed-point map at the current iterate.
            la::Vector<T> f_k(_G_hist.front().index_map(),
                _G_hist.front().bs());
            for (int i = 0; i < n; ++i)
                f_k[i] = G_k[i] - x_k[i];

            // Normal equations F^T F alpha = F^T f_k over the history.
            // m_k is O(5), so the m_k x m_k system is solved by dense
            // column-pivoted QR (avoiding a full N x m_k factorization).
            Eigen::MatrixXd FtF(m_k, m_k);
            Eigen::VectorXd Ftf(m_k);
            for (int i = 0; i < m_k; ++i) {
                const la::Vector<T>& Gi = _G_hist[i];
                const la::Vector<T>& xi = _x_hist[i];
                double dot = 0;
                for (int j = 0; j < n; ++j) {
                    const double fi_j = (G_k[j] - x_k[j])
                        - (Gi[j] - xi[j]);
                    dot += fi_j * (G_k[j] - x_k[j]);
                }
                Ftf(i) = dot;
                for (int l = 0; l <= i; ++l) {
                    const la::Vector<T>& Gl = _G_hist[l];
                    const la::Vector<T>& xl = _x_hist[l];
                    double s = 0;
                    for (int j = 0; j < n; ++j) {
                        const double fi_j = (G_k[j] - x_k[j])
                            - (Gi[j] - xi[j]);
                        const double fl_j = (G_k[j] - x_k[j])
                            - (Gl[j] - xl[j]);
                        s += fi_j * fl_j;
                    }
                    FtF(i, l) = s;
                    FtF(l, i) = s;
                }
            }

            Eigen::VectorXd alpha
                = FtF.colPivHouseholderQr().solve(Ftf);
            if (_dampening < 1.0)
                alpha *= _dampening;

            // x_prop = (1 - sum(alpha)) G_k + sum_j alpha(j) G_hist[j].
            la::Vector<T> x_prop(_G_hist.front().index_map(),
                _G_hist.front().bs());
            const double a_sum = alpha.sum();
            for (int j = 0; j < n; ++j)
                x_prop[j] = (1.0 - a_sum) * G_k[j];
            for (int i = 0; i < m_k; ++i)
                for (int j = 0; j < n; ++j)
                    x_prop[j] += alpha(i) * _G_hist[i][j];

            // Divergence guard: compare infinity norms of the proposed
            // step and the plain Picard step.
            if (_max_growth > 0) {
                double prop_norm = 0, naive_norm = 0;
                for (int j = 0; j < n; ++j) {
                    prop_norm = std::max(prop_norm,
                        static_cast<double>(std::abs(x_prop[j] - x_k[j])));
                    naive_norm = std::max(naive_norm,
                        static_cast<double>(std::abs(G_k[j] - x_k[j])));
                }
                if (naive_norm > 0 and prop_norm > _max_growth * naive_norm) {
                    if (_reset_on_growth) {
                        _x_hist.clear();
                        _G_hist.clear();
                    }
                    return std::nullopt;
                }
            }
            return x_prop;
        }

        /// Number of (x, G) pairs pushed so far.
        int iteration_count() const { return _iter_count; }

        /// Whether the history is non-empty.
        bool has_history() const { return not _G_hist.empty(); }

    private:
        int _depth;
        int _warmup_iters;
        double _dampening;
        double _max_growth;
        bool _reset_on_growth;

        // History; index 0 is the most recent pair. A deque keeps
        // push_front O(1), as push happens every iteration. Mutable so
        // `step` can reset the history when the divergence guard trips.
        mutable std::deque<la::Vector<T>> _x_hist;
        mutable std::deque<la::Vector<T>> _G_hist;
        int _iter_count = 0;
    };

    /// Solve a nonlinear system by Anderson-accelerated Picard
    /// iteration on the fixed-point map `x <- G(x)`.
    ///
    /// At each iterate the caller provides the linear system whose
    /// solution is `G(x)` (the fixed-point map); the iteration mixes the
    /// last `depth` iterates to accelerate the plain Picard
    /// `x <- x + omega (G(x) - x)`.
    ///
    /// @param[in] system_fn Builds, at the current iterate `x`, the linear
    ///   system `A G = b` whose solution is `G(x)`.
    /// @param[in,out] x Initial guess on entry, solution on exit.
    /// @param[in] cfg Configuration.
    /// @return Convergence status and iteration counts.
    template <std::floating_point T>
    AndersonResult<T> anderson_picard(
        const std::function<std::pair<la::MatrixCSR<T>, la::Vector<T>>(
            const la::Vector<T>&)>& system_fn,
        la::Vector<T>& x, const AndersonConfig& cfg = {})
    {
        AndersonMixer<T> mixer(cfg.depth, cfg.warmup_iters, cfg.dampening,
            cfg.max_growth, cfg.reset_on_growth);

        AndersonResult<T> result;
        la::Vector<T> G(x.index_map(), x.bs());
        const int n = static_cast<int>(x.array().size());
        const T one = T(1);

        for (int it = 0; it < cfg.max_iterations; ++it) {
            // Build the frozen linear system A G = b and its residual.
            auto [A, b] = system_fn(x);
            la::Vector<T> r(x.index_map(), x.bs());
            r.set(0);
            A.mult(x, r); // r += A x
            double max_residual = 0, max_b = 0;
            for (int i = 0; i < n; ++i) {
                max_residual = std::max(
                    max_residual, static_cast<double>(std::abs(b[i] - r[i])));
                max_b = std::max(
                    max_b, static_cast<double>(std::abs(b[i])));
            }
            const double residual_threshold
                = cfg.relative_tolerance * max_b + cfg.absolute_tolerance;

            // Early exit on the residual alone.
            if (it > 0 and max_residual <= residual_threshold) {
                result.converged = true;
                result.iterations = it;
                return result;
            }

            // G(x) = solve(A, b). The Krylov solver starts from a zero
            // initial guess (set_initial_guess is false by default), so the
            // reused vector must be cleared each solve.
            G.set(0);
            la::KrylovSolver<T> ks;
            ks.set_operator(A); // copies; enables preconditioner synthesis
            if (cfg.preconditioner_type != "none")
                ks.set_preconditioner_type(cfg.preconditioner_type);
            ks.set_solver_type(cfg.linear_solver_type);
            ks.set_tolerances(
                cfg.krylov_rtol, cfg.krylov_atol, cfg.krylov_max_iter);
            const int k_it = ks.solve(G, b);
            result.krylov_iterations += k_it;
            if (k_it >= cfg.krylov_max_iter)
                throw std::runtime_error(
                    "anderson_picard: inner linear solve did not converge.");
            cfg.post_linear_solve();

            // Mix the iterate.
            std::optional<la::Vector<T>> prop = mixer.step(x, G);
            la::Vector<T> next(x.index_map(), x.bs());
            if (prop) {
                bool finite = true;
                for (int i = 0; i < n; ++i)
                    if (not std::isfinite((*prop)[i])) {
                        finite = false;
                        break;
                    }
                if (finite) {
                    next = *prop;
                }
                else {
                    // Non-finite mixed step: fall back to damped Picard.
                    for (int i = 0; i < n; ++i)
                        next[i] = x[i] + one * cfg.dampening * (G[i] - x[i]);
                }
            }
            else {
                for (int i = 0; i < n; ++i)
                    next[i] = x[i] + one * cfg.dampening * (G[i] - x[i]);
            }

            // Convergence test on both the update and the residual.
            double max_update = 0, max_state = 0;
            for (int i = 0; i < n; ++i) {
                max_update = std::max(
                    max_update, static_cast<double>(std::abs(next[i] - x[i])));
                max_state = std::max(
                    max_state, static_cast<double>(std::abs(next[i])));
            }
            const double update_threshold
                = cfg.relative_tolerance * max_state + cfg.absolute_tolerance;
            if (max_update <= update_threshold
                and max_residual <= residual_threshold) {
                x = next;
                result.converged = true;
                result.iterations = it + 1;
                return result;
            }

            if (cfg.report)
                spdlog::info("Anderson Picard iteration {}: residual = {:.3e}",
                    it + 1, max_residual);

            // Record the pre-mix pair (x_k, G_k); the mixing uses the
            // history of past iterates, so it must not include `next`.
            mixer.push(x, G);
            x = next;
        }

        if (cfg.report)
            spdlog::warn("anderson_picard did not converge in {} iterations.",
                cfg.max_iterations);
        result.iterations = cfg.max_iterations;
        return result;
    }

} // namespace hellofem::nls
