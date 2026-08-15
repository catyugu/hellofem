// hellofem::nls — Newton-Krylov (JFNK) nonlinear solver
// SPDX-License-Identifier: MIT

#pragma once

#include "la/KrylovSolver.h"
#include "la/LinearOperator.h"
#include "la/MatrixCSR.h"
#include "la/Vector.h"
#include "la/preconditioner.h"

#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include <spdlog/spdlog.h>

namespace hellofem::nls {

    /// Newton-Krylov solver for `F(x) = 0`.
    ///
    /// Two modes, selected by which callbacks are registered:
    /// - **Assembled Jacobian**: `set_jacobian` registers a callback that
    ///   assembles the Jacobian matrix at `x` into a caller-owned matrix.
    ///   The Newton updates solve that matrix with a Krylov method.
    /// - **Matrix-free JFNK**: without `set_jacobian`, the solver builds a
    ///   finite-difference Jacobian-vector product
    ///   `J v ~= (F(x + step v) - F(x)) / step` from the residual callback
    ///   and feeds it directly to the Krylov solver (no matrix is ever
    ///   formed). `step = sqrt(eps) * max(|x|, 1) / |v|` follows the
    ///   standard Knoll-Keyser scaling so the perturbation size tracks the
    ///   local solution scale.
    ///
    /// Residual and Jacobian callbacks hold caller-owned scratch (a vector
    /// for the residual, a matrix for the Jacobian), mirroring the dolfinx
    /// NewtonSolver contract but with matrices replaced by `la::Vector` /
    /// `la::MatrixCSR` and MPI removed.
    ///
    /// @tparam T Scalar type (real or complex).
    template <std::floating_point T>
    class NewtonSolver {
    public:
        /// Value type.
        using value_type = T;

        /// Residual callback: `b = F(x)`.
        using residual_fn = std::function<void(const la::Vector<T>& x,
            la::Vector<T>& b)>;

        /// Jacobian callback: assemble `J = dF/dx` at `x`.
        using jacobian_fn = std::function<void(const la::Vector<T>& x,
            la::MatrixCSR<T>& J)>;

        /// Preconditioner callback: build `P` (approximate inverse of the
        /// Jacobian) at `x`. Optional; without it the inner solve is
        /// unpreconditioned (or, in assembled mode, uses
        /// `preconditioner_type`).
        using precond_fn = std::function<void(const la::Vector<T>& x,
            std::shared_ptr<const la::Preconditioner<T>>& P)>;

        /// Default constructor.
        NewtonSolver() = default;

        /// Register the residual callback `b = F(x)`.
        /// @param[in] F The residual function.
        /// @param[in,out] b Caller-owned scratch that `F` fills. The solver
        ///   reads it for convergence and FD differencing.
        void set_residual(residual_fn F, la::Vector<T>& b)
        {
            _fnF = std::move(F);
            _b = &b;
            _fd_scratch = std::make_unique<la::Vector<T>>(
                b.index_map(), b.bs());
        }

        /// Register the Jacobian callback. If set, the solver assembles a
        /// real matrix each Newton step (assembled mode). If unset, the
        /// solver uses a finite-difference Jacobian-vector product (JFNK).
        /// @param[in] J The Jacobian function.
        /// @param[in,out] Jmat Caller-owned matrix that `J` fills.
        void set_jacobian(jacobian_fn J, la::MatrixCSR<T>& Jmat)
        {
            _fnJ = std::move(J);
            _Jmat = &Jmat;
        }

        /// Register an optional preconditioner callback `P(x) -> P`.
        void set_preconditioner(precond_fn P) { _fnP = std::move(P); }

        /// Maximum number of Newton iterations.
        int max_it = 50;

        /// Relative residual convergence tolerance.
        double rtol = 1e-9;

        /// Absolute residual convergence tolerance.
        double atol = 1e-10;

        /// Convergence criterion: "residual" tests the residual norm,
        /// "incremental" tests the Newton update norm.
        std::string convergence_criterion = "residual";

        /// Damping on the Newton update: `x -= relaxation_parameter * dx`.
        double relaxation_parameter = 1.0;

        /// Log each iteration.
        bool report = true;

        /// Throw if the Newton iteration does not converge.
        bool error_on_nonconvergence = true;

        /// Inner linear solver type ("gmres" or "cg").
        std::string linear_solver = "gmres";

        /// Inner linear solver tolerances / iteration cap.
        double linear_rtol = 1e-6;
        double linear_atol = 1e-12;
        int linear_max_iter = 1000;

        /// Preconditioner synthesized from the assembled Jacobian each
        /// step ("none", "jacobi", "amg"). Ignored in matrix-free mode
        /// (use `set_preconditioner` there).
        std::string preconditioner_type = "none";

        /// Solve `F(x) = 0` from the initial guess `x`.
        /// @param[in,out] x Initial guess on entry, solution on exit.
        /// @return (Newton iterations used, converged).
        std::pair<int, bool> solve(la::Vector<T>& x)
        {
            if (!_fnF)
                throw std::runtime_error(
                    "NewtonSolver: residual callback not set.");
            if (_fnJ and !_Jmat)
                throw std::runtime_error(
                    "NewtonSolver: Jacobian matrix not set.");

            _iteration = 0;
            _krylov_iterations = 0;
            _residual = -1;
            _residual0 = 0;

            la::Vector<T> dx(_b->index_map(), _b->bs());

            // Initial residual F(x).
            _fnF(x, *_b);
            _residual0 = la::norm(*_b, la::Norm::l2);
            _residual = _residual0;

            // Convergence pre-check; "incremental" needs one update first.
            bool converged = false;
            if (convergence_criterion == "incremental") {
                converged = false;
            }
            else if (convergence_criterion == "residual") {
                converged = _converged(*_b);
            }
            else {
                throw std::runtime_error(
                    "NewtonSolver: unknown convergence criterion '"
                    + convergence_criterion + "'.");
            }

            while (!converged and _iteration < max_it) {
                if (_fnJ) {
                    // Assembled mode: build the Jacobian matrix.
                    _fnJ(x, *_Jmat);
                }

                // Configure the inner Krylov solve.
                la::KrylovSolver<T> ks;
                if (_fnJ) {
                    ks.set_operator(*_Jmat); // copies; enables synth. precond
                    if (preconditioner_type != "none")
                        ks.set_preconditioner_type(preconditioner_type);
                }
                else {
                    // JFNK: finite-difference Jacobian-vector product.
                    ks.set_operator(_fd_operator(x, *_b));
                }
                if (_fnP) {
                    std::shared_ptr<const la::Preconditioner<T>> P;
                    _fnP(x, P);
                    ks.set_preconditioner(P);
                }
                ks.set_solver_type(linear_solver);
                ks.set_tolerances(linear_rtol, linear_atol, linear_max_iter);

                // Newton update: solve J dx = F(x).
                dx.set(0);
                const int krylov_iters = ks.solve(dx, *_b);
                _krylov_iterations += krylov_iters;
                if (krylov_iters >= linear_max_iter) {
                    // Inner solve did not converge.
                    if (error_on_nonconvergence)
                        throw std::runtime_error(
                            "NewtonSolver: inner linear solve did not "
                            "converge.");
                    spdlog::warn(
                        "NewtonSolver: inner linear solve did not converge.");
                    return {_iteration, false};
                }

                // x <- x - relaxation * dx.
                for (std::size_t i = 0; i < x.array().size(); ++i)
                    x[i] -= relaxation_parameter * dx[i];

                ++_iteration;

                // Recompute the residual at the new iterate.
                _fnF(x, *_b);
                _residual = la::norm(*_b, la::Norm::l2);

                if (report)
                    spdlog::info("Newton iteration {}: r (abs) = {:.3e} "
                                 "(tol = {:.1e}), r0 = {:.3e}",
                        _iteration, _residual, atol, _residual0);

                if (convergence_criterion == "incremental") {
                    const double inc = la::norm(dx, la::Norm::l2);
                    const double thresh
                        = rtol * _residual0 + atol;
                    if (inc < thresh)
                        converged = true;
                }
                else {
                    converged = _converged(*_b);
                }
            }

            if (converged) {
                if (report)
                    spdlog::info(
                        "NewtonSolver finished in {} iterations and {} "
                        "linear solves.",
                        _iteration, _krylov_iterations);
                return {_iteration, true};
            }

            if (error_on_nonconvergence)
                throw std::runtime_error(
                    "NewtonSolver did not converge: maximum number of "
                    "iterations reached.");
            spdlog::warn("NewtonSolver did not converge in {} iterations.",
                _iteration);
            return {_iteration, false};
        }

        /// Number of Newton iterations performed.
        int iteration() const { return _iteration; }

        /// Number of inner (Krylov) iterations accumulated.
        int krylov_iterations() const { return _krylov_iterations; }

        /// Current residual norm.
        double residual() const { return _residual; }

        /// Initial residual norm.
        double residual0() const { return _residual0; }

    private:
        /// Convergence test on the residual: relative to `_residual0` or
        /// absolute.
        bool _converged(const la::Vector<T>& b) const
        {
            const double r = la::norm(b, la::Norm::l2);
            return r < atol or r < rtol * _residual0;
        }

        /// Build a linear operator `y += J v` where `J` is the
        /// finite-difference Jacobian of `F` at `x`.
        ///
        /// `J v ~= (F(x + step v) - F(x)) / step` with
        /// `step = sqrt(eps) * max(|x|, 1) / |v|`. The baseline residual
        /// `F(x)` is snapshotted by value. Perturbed residuals are written
        /// into a dedicated scratch vector, never into the solver's
        /// right-hand side (which the Krylov solver reads).
        la::LinearOperator<T> _fd_operator(const la::Vector<T>& x,
            const la::Vector<T>& b)
        {
            const la::Vector<T> b0 = b;
            return la::LinearOperator<T>(
                [&, this, b0](const la::Vector<T>& v, la::Vector<T>& y) {
                    const T vn = la::norm(v, la::Norm::l2);
                    if (vn == 0)
                        return; // v = 0 -> J v = 0

                    T xscale = 0;
                    for (const T xi : x.array())
                        xscale = std::max(xscale, std::abs(xi));
                    const T step = std::sqrt(std::numeric_limits<T>::epsilon())
                        * (xscale > 1 ? xscale : T(1))
                        / vn;

                    la::Vector<T> xp(_b->index_map(), _b->bs());
                    for (std::size_t i = 0; i < x.array().size(); ++i)
                        xp[i] = x[i] + step * v[i];
                    _fnF(xp, *_fd_scratch);
                    for (std::size_t i = 0; i < y.array().size(); ++i)
                        y[i] += ((*_fd_scratch)[i] - b0[i]) / step;
                });
        }

        // Callbacks and scratch (caller-owned).
        residual_fn _fnF;
        la::Vector<T>* _b = nullptr;
        jacobian_fn _fnJ;
        la::MatrixCSR<T>* _Jmat = nullptr;
        precond_fn _fnP;

        // Scratch for perturbed residuals in the matrix-free operator.
        std::unique_ptr<la::Vector<T>> _fd_scratch;

        // Iteration counters.
        int _iteration = 0;
        int _krylov_iterations = 0;
        double _residual = -1;
        double _residual0 = 0;
    };

} // namespace hellofem::nls
