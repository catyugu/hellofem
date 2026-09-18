// hellofem::nls — Anderson-accelerated Picard (fixed-point) iteration
// SPDX-License-Identifier: MIT

#pragma once

#include "la/LinearSolver.h"
#include "la/MatrixCSR.h"
#include "la/Vector.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <deque>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <spdlog/spdlog.h>

namespace hellofem::nls {

    /// Settings of the Anderson-accelerated Picard iteration, named after
    /// the COMSOL nonlinear solver fields they mirror. The defaults are
    /// COMSOL's.
    struct AndersonConfig {
        /// "Dimension of iteration space": number of past iterates kept for
        /// the mixing.
        int dimension = 5;

        /// "Mixing parameter": under-relaxation applied to every step.
        double mixing = 0.9;

        /// "Iteration delay": iterations taken before the mixing starts.
        /// The history is recorded from the first iteration on, so the
        /// mixing has the iterates of the delay to work with when it starts.
        int delay = 0;

        /// "Threshold for Anderson step": the mixed step is taken while its
        /// infinity norm stays below this multiple of the previous step's,
        /// and the plain step is taken past it. Lowering it trades speed for
        /// robustness.
        double threshold = 10.0;

        /// Relative tolerance on the residual of the frozen system.
        double relative_tolerance = 1e-8;

        /// Absolute tolerance on the residual of the frozen system.
        double absolute_tolerance = 1e-12;

        /// Maximum number of iterations.
        int max_iterations = 50;

        /// Log each iteration.
        bool report = true;

        /// Settings of the inner solve of the frozen system `A G = b`.
        la::LinearSettings linear;

        /// Seed the inner solve with the current iterate. The frozen system
        /// moves little between iterations, so this cuts the Krylov work; it
        /// does not affect the converged iterate.
        bool warm_start_linear_solve = true;
    };

    /// Result of an Anderson-accelerated Picard solve.
    template <std::floating_point T>
    struct AndersonResult {
        /// Whether the iteration converged.
        bool converged = false;

        /// Number of iterations used.
        int iterations = 0;

        /// Number of inner linear-solver iterations accumulated.
        int krylov_iterations = 0;
    };

    /// Anderson mixing of the fixed-point iteration `x <- G(x)`.
    ///
    /// With `f = G(x) - x` and, over the history, `df_i = f - f_i`, the step
    /// the mixer returns is
    ///
    ///     s = f - sum_i alpha_i (G - G_i),
    ///     alpha = argmin_alpha || f - sum_i alpha_i df_i ||_2 .
    ///
    /// Anderson's mixing writes the next iterate as `sum_i a_i G_i` with the
    /// coefficients summing to one; that is this least-squares problem in the
    /// residuals `f_i`, and `G - G_i = df_i + (x - x_i)` is what leaves the
    /// plain `f` in the step.
    ///
    /// The mixing is applied to the step rather than to the iterate, so the
    /// mixing parameter is a plain under-relaxation of it:
    /// `x += mixing * step(x, G)`.
    ///
    /// @tparam T Scalar type.
    template <std::floating_point T>
    class AndersonMixer {
    public:
        /// Vector type of the iterate.
        using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;

        /// Create a mixer.
        /// @param[in] cfg Configuration of the mixing.
        explicit AndersonMixer(const AndersonConfig& cfg)
            : _dimension(std::max(cfg.dimension, 0)), _delay(std::max(cfg.delay, 0)),
              _threshold(cfg.threshold)
        {
        }

        /// Step to add to the iterate `x` for its image `G`, recording the
        /// pair in the history. The caller scales the step by the mixing
        /// parameter: `x += mixing * step(x, G)`.
        /// @param[in] x Current iterate.
        /// @param[in] G Fixed-point image `G(x)`.
        /// @return The step to add to `x`.
        Vec step(const Eigen::Ref<const Vec>& x, const Eigen::Ref<const Vec>& G)
        {
            const Vec f = G - x;
            const int m = _iterations < _delay
                ? 0
                : std::min(_dimension, static_cast<int>(_G_hist.size()));

            Vec s = f;
            if (m > 0) {
                const Vec mixed = _mixed(G, f, m);
                const double norm = static_cast<double>(
                    mixed.template lpNorm<Eigen::Infinity>());
                // A mixed step far longer than the one the previous iteration
                // took, or a non-finite one out of a history the
                // least-squares solve cannot determine, is replaced by the
                // plain step.
                if (std::isfinite(norm) and norm <= _threshold * _prev_step)
                    s = mixed;
            }
            _prev_step = static_cast<double>(
                s.template lpNorm<Eigen::Infinity>());
            _push(f, G);
            ++_iterations;
            return s;
        }

        /// Number of pairs in the history.
        int history_size() const { return static_cast<int>(_G_hist.size()); }

    private:
        /// The mixed step `f - sum_i alpha_i (G - G_i)`.
        Vec _mixed(const Eigen::Ref<const Vec>& G, const Vec& f, int m)
        {
            if (static_cast<int>(_df.size()) < m)
                _df.resize(static_cast<std::size_t>(m), Vec(f.size()));

            // The residual differences against the history, and the normal
            // equations `F^T F alpha = F^T f` of the mixing coefficients.
            // `m` is the dimension of the iteration space, a handful of
            // vectors, so the small dense system is solved directly.
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> gram(m, m);
            Eigen::Matrix<T, Eigen::Dynamic, 1> rhs(m);
            for (int i = 0; i < m; ++i) {
                _df[i] = f - _f_hist[i]; // df_i = f - f_i
                rhs(i) = _df[i].dot(f);
                gram(i, i) = _df[i].squaredNorm();
                for (int j = 0; j < i; ++j)
                    gram(i, j) = gram(j, i) = _df[i].dot(_df[j]);
            }
            const Eigen::Matrix<T, Eigen::Dynamic, 1> alpha
                = gram.colPivHouseholderQr().solve(rhs);

            Vec s = f;
            for (int i = 0; i < m; ++i)
                s.noalias() -= alpha(i) * (G - _G_hist[i]);
            return s;
        }

        /// Record `(f, G)`, keeping at most `dimension` pairs.
        void _push(const Vec& f, const Eigen::Ref<const Vec>& G)
        {
            if (_dimension == 0)
                return;
            _f_hist.emplace_front(f);
            _G_hist.emplace_front(G);
            if (static_cast<int>(_G_hist.size()) > _dimension) {
                _f_hist.pop_back();
                _G_hist.pop_back();
            }
        }

        int _dimension;
        int _delay;
        double _threshold;

        // History; index 0 is the most recent pair. A deque keeps the
        // push_front of every iteration O(1). The pair is kept as the
        // residual `f = G - x` and the image `G`: the mixing needs both, and
        // forming `f` once per iteration rather than once per history entry
        // per mixing step is what keeps the history free of the iterate.
        std::deque<Vec> _f_hist;
        std::deque<Vec> _G_hist;

        // Scratch for the residual differences of one mixing step, one per
        // history entry.
        std::vector<Vec> _df;

        // Infinity norm of the step the last call returned: the reference the
        // threshold for the Anderson step is measured against.
        double _prev_step = 0;

        // Iterations stepped through, the iteration delay being counted in
        // those.
        int _iterations = 0;
    };

    /// Solve a nonlinear system by Anderson-accelerated Picard iteration on
    /// the fixed-point map `x <- G(x)`.
    ///
    /// One iteration builds the frozen system `A(x) G = b(x)`, solves it for
    /// `G(x)`, and takes the step the mixing proposes from it. The iteration
    /// converges when the residual of the frozen system at the iterate is
    /// below `relative_tolerance * max|b| + absolute_tolerance`; an initial
    /// guess that already meets it is returned without a solve.
    ///
    /// @param[in] system_fn Builds, at the current iterate `x`, the linear
    ///   system `A G = b` whose solution is `G(x)`.
    /// @param[in,out] x Initial guess on entry, solution on exit.
    /// @param[in,out] inner The solve of the frozen system. It belongs to the
    ///   caller, so that the factorization of an operator outlives one call:
    ///   an iteration that re-assembles the operator it solved before finds
    ///   the factorization already built (see `la::LinearSolver`).
    /// @param[in] cfg Configuration.
    /// @return Convergence status and iteration counts.
    template <std::floating_point T>
    AndersonResult<T> anderson_picard(
        const std::function<std::pair<la::MatrixCSR<T>, la::Vector<T>>(
            const la::Vector<T>&)>& system_fn,
        la::Vector<T>& x, la::LinearSolver<T>& inner,
        const AndersonConfig& cfg = {})
    {
        using Vec = typename AndersonMixer<T>::Vec;
        AndersonResult<T> result;
        AndersonMixer<T> mixer(cfg);

        const Eigen::Index n = static_cast<Eigen::Index>(x.array().size());
        Eigen::Map<Vec> xv(x.array().data(), n);
        la::Vector<T> G(x.index_map(), x.bs());
        la::Vector<T> r(x.index_map(), x.bs());
        Eigen::Map<Vec> gv(G.array().data(), n);
        Eigen::Map<Vec> rv(r.array().data(), n);

        for (int it = 0; it < cfg.max_iterations; ++it) {
            // The frozen system at the current iterate, and the residual of
            // the iterate in it: the measure the iteration converges on, so
            // it is taken before the solve it decides on.
            auto [A, b] = system_fn(x);
            const Eigen::Map<const Vec> bv(b.array().data(), n);
            r.set(0);
            A.mult(x, r);
            rv = bv - rv;
            const double residual = static_cast<double>(
                rv.template lpNorm<Eigen::Infinity>());
            const double scale = static_cast<double>(
                bv.template lpNorm<Eigen::Infinity>());
            const double threshold
                = cfg.relative_tolerance * scale + cfg.absolute_tolerance;
            if (residual <= threshold) {
                result.converged = true;
                result.iterations = it;
                return result;
            }

            // G(x): the solution of the frozen system, from the current
            // iterate when a warm start is asked for.
            if (cfg.warm_start_linear_solve)
                gv = xv;
            else
                gv.setZero();
            const int krylov = inner.solve(
                A, G, b, cfg.warm_start_linear_solve, cfg.linear);
            result.krylov_iterations += krylov;
            if (krylov >= cfg.linear.max_iterations)
                throw std::runtime_error(
                    "anderson_picard: the inner linear solve did not "
                    "converge.");

            // The accuracy the inner solve delivered bounds the residual the
            // iteration can reach. The inner tolerance is relative to the
            // 2-norm of the right-hand side and the outer threshold to its
            // infinity norm, so an inner tolerance looser than the outer
            // threshold leaves a floor the iteration can never get under: it
            // would spend every remaining iteration on a fixed point that
            // does not move.
            r.set(0);
            A.mult(G, r);
            rv = bv - rv;
            const double inner_residual = static_cast<double>(
                rv.template lpNorm<Eigen::Infinity>());
            if (inner_residual > threshold and residual <= inner_residual)
                throw std::runtime_error(
                    "anderson_picard: the inner solve leaves a residual of "
                    + std::to_string(inner_residual) + ", above the nonlinear "
                    "threshold of "
                    + std::to_string(threshold)
                    + ": tighten the inner tolerance (linear.rtol / "
                      "linear.atol).");

            xv.noalias() += T(cfg.mixing) * mixer.step(xv, gv);

            if (cfg.report)
                spdlog::info("Anderson Picard iteration {}: residual = {:.3e}",
                    it + 1, residual);
        }

        if (cfg.report)
            spdlog::warn("anderson_picard did not converge in {} iterations.",
                cfg.max_iterations);
        result.iterations = cfg.max_iterations;
        return result;
    }

} // namespace hellofem::nls
