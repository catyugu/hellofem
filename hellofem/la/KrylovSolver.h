// hellofem::la — Krylov subspace iterative solvers
// SPDX-License-Identifier: MIT

#pragma once

#include "LinearOperator.h"
#include "preconditioner.h"
#include "schwarz.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <spdlog/spdlog.h>

namespace hellofem::la {

    namespace impl {
        /// Conjugate, identity for real scalars.
        template <typename T>
        T conj_if_complex(const T& v)
        {
            if constexpr (std::is_same<T, std::complex<double>>::value
                or std::is_same<T, std::complex<float>>::value)
                return std::conj(v);
            else
                return v;
        }
    } // namespace impl

    /// Iterative Krylov solver for `A x = b`.
    ///
    /// Acts on a `LinearOperator` (an assembled `MatrixCSR` or a
    /// matrix-free operator), optionally preconditioned with a
    /// `Preconditioner`. Solver and preconditioner types are configured
    /// with first-class C++ settings (replacing dolfinx's PETSc options).
    ///
    /// @tparam T Scalar type of the system.
    template <typename T>
    class KrylovSolver {
    public:
        /// Solver type.
        enum class SolverType {
            cg, ///< Conjugate gradients (SPD systems)
            gmres, ///< Restarted GMRES (general systems)
            bicgstab, ///< BiCGSTAB (general/non-symmetric systems)
        };

        /// Default constructor
        KrylovSolver() = default;

        /// Set the operator (matrix-free or assembled).
        void set_operator(LinearOperator<T> A) { _A = std::move(A); }

        /// Set the operator from an assembled matrix. The matrix is
        /// copied so that `set_preconditioner_type` can synthesize a
        /// Jacobi/AMG/ILU/Schwarz preconditioner from it.
        void set_operator(const MatrixCSR<T>& A)
        {
            _A = A.as_operator();
            _matrix = std::make_shared<const MatrixCSR<T>>(A);
        }

        /// Register a matrix for preconditioner synthesis without changing
        /// the operator (used for matrix-free solves with a matrix-based
        /// preconditioner, e.g. JFNK with an assembled Jacobian).
        void set_matrix(const MatrixCSR<T>& A)
        {
            _matrix = std::make_shared<const MatrixCSR<T>>(A);
        }

        /// Set the operator and preconditioner together.
        void set_operators(LinearOperator<T> A,
            std::shared_ptr<const Preconditioner<T>> P)
        {
            _A = std::move(A);
            _P = std::move(P);
        }

        /// Set the preconditioner explicitly.
        void set_preconditioner(std::shared_ptr<const Preconditioner<T>> P)
        {
            _P = std::move(P);
        }

        /// Configure the solver type by name.
        /// @param[in] type "cg", "gmres" or "bicgstab".
        void set_solver_type(std::string_view type)
        {
            if (type == "cg")
                _type = SolverType::cg;
            else if (type == "gmres")
                _type = SolverType::gmres;
            else if (type == "bicgstab")
                _type = SolverType::bicgstab;
            else
                throw std::runtime_error("Unknown solver type.");
        }

        /// Configure the preconditioner by name.
        /// @param[in] type "none", "jacobi", "ilu" or "amg". The latter
        /// three require a matrix registered via `set_operator(MatrixCSR)`.
        void set_preconditioner_type(std::string_view type)
        {
            if (type == "none")
                _P.reset();
            else if (type == "jacobi") {
                if (!_matrix)
                    throw std::runtime_error("Jacobi preconditioner requires "
                                             "a matrix (set_operator).");
                _P = std::make_shared<JacobiPreconditioner<T>>(*_matrix);
            }
            else if (type == "amg") {
                if (!_matrix)
                    throw std::runtime_error("AMG preconditioner requires a "
                                             "matrix (set_operator).");
                _P = std::make_shared<AmgPreconditioner<T>>(*_matrix);
            }
            else if (type == "ilu") {
                if (!_matrix)
                    throw std::runtime_error("ILU preconditioner requires a "
                                             "matrix (set_operator).");
                _P = std::make_shared<IluPreconditioner<T>>(*_matrix);
            }
            else if (type == "schwarz") {
                if (!_matrix)
                    throw std::runtime_error("Schwarz preconditioner requires "
                                             "a matrix (set_operator).");
                _P = std::make_shared<SchwarzPreconditioner<T>>(*_matrix, 2, 1);
            }
            else
                throw std::runtime_error("Unknown preconditioner type.");
        }

        /// Set convergence tolerances.
        /// @param[in] rtol Relative tolerance on the residual norm.
        /// @param[in] atol Absolute tolerance on the residual norm.
        /// @param[in] max_iter Maximum number of iterations.
        void set_tolerances(double rtol, double atol, int max_iter)
        {
            _rtol = rtol;
            _atol = atol;
            _max_iter = max_iter;
        }

        /// Control whether the initial guess in `solve` is used.
        void set_initial_guess(bool use) { _use_initial_guess = use; }

        /// Solve `A x = b`.
        ///
        /// @param[in,out] x On entry the initial guess (if
        /// `set_initial_guess(true)`), on exit the solution.
        /// @param[in] b The right-hand side.
        /// @param[in] transpose Unused (neither CG nor GMRES uses A^T).
        /// @return The number of iterations used. On non-convergence
        /// within `max_iter` iterations, a warning is logged and the
        /// iteration count reached is returned (no exception).
        int solve(Vector<T>& x, const Vector<T>& b, bool transpose = false) const
        {
            (void)transpose; // CG/GMRES/BiCGSTAB do not use the transpose
            switch (_type) {
            case SolverType::cg:
                return _cg(x, b);
            case SolverType::bicgstab:
                return _bicgstab(x, b);
            default:
                return _gmres(x, b);
            }
        }

    private:
        /// Preconditioned conjugate gradient method.
        int _cg(Vector<T>& x, const Vector<T>& b) const
        {
            const std::size_t n = b.array().size();
            Vector<T> r(b.index_map(), b.bs());
            Vector<T> z(b.index_map(), b.bs());
            Vector<T> p(b.index_map(), b.bs());
            Vector<T> q(b.index_map(), b.bs());

            r = b;
            if (_use_initial_guess) {
                q.set(0);
                _A.mult(x, q);
                for (std::size_t i = 0; i < n; ++i)
                    r[i] -= q[i];
            }

            _apply_preconditioner(r, z);
            p = z;
            T rho = inner_product(r, z);
            const auto tol
                = std::max(static_cast<decltype(squared_norm(b))>(_atol),
                    static_cast<decltype(squared_norm(b))>(_rtol)
                        * std::sqrt(squared_norm(b)));

            for (int k = 0; k < _max_iter; ++k) {
                q.set(0);
                _A.mult(p, q);
                const T pq = inner_product(p, q);
                const T alpha = rho / pq;

                for (std::size_t i = 0; i < n; ++i) {
                    x[i] += alpha * p[i];
                    r[i] -= alpha * q[i];
                }

                if (std::sqrt(squared_norm(r)) <= tol)
                    return k + 1;

                _apply_preconditioner(r, z);
                const T rho_new = inner_product(r, z);
                const T beta = rho_new / rho;
                rho = rho_new;
                for (std::size_t i = 0; i < n; ++i)
                    p[i] = z[i] + beta * p[i];
            }

            spdlog::warn("CG did not converge in {} iterations.", _max_iter);
            return _max_iter;
        }

        /// Restarted (m=30) right-preconditioned GMRES.
        int _gmres(Vector<T>& x, const Vector<T>& b) const
        {
            const std::size_t n = b.array().size();
            constexpr int m = 30;

            Vector<T> r(b.index_map(), b.bs());
            r = b;
            if (_use_initial_guess) {
                Vector<T> ax(b.index_map(), b.bs());
                ax.set(0);
                _A.mult(x, ax);
                for (std::size_t i = 0; i < n; ++i)
                    r[i] -= ax[i];
            }

            // Preconditioned operator: w = A P v
            auto op = [&](const Vector<T>& v, Vector<T>& w) {
                Vector<T> z(b.index_map(), b.bs());
                z.set(0);
                _apply_preconditioner(v, z);
                w.set(0);
                _A.mult(z, w);
            };

            const auto tol = std::max(static_cast<decltype(squared_norm(b))>(_atol),
                static_cast<decltype(squared_norm(b))>(_rtol)
                    * std::sqrt(squared_norm(b)));

            int total_iter = 0;
            for (int outer = 0; outer < _max_iter / m; ++outer) {
                const T beta = std::sqrt(squared_norm(r));
                if (beta <= tol)
                    return total_iter;

                std::vector<Vector<T>> V;
                V.reserve(m + 1);
                V.emplace_back(b.index_map(), b.bs());
                V[0].set(0);
                for (std::size_t i = 0; i < n; ++i)
                    V[0][i] = r[i] / beta;

                // Hessenberg and Givens rotations
                std::vector<std::vector<T>> H(m + 1, std::vector<T>(m, 0));
                std::vector<T> gs(m + 1, 0);
                gs[0] = beta;
                std::vector<T> c(m, 0), s(m, 0);

                int k;
                for (k = 0; k < m and total_iter < _max_iter; ++k, ++total_iter) {
                    V.emplace_back(b.index_map(), b.bs());
                    Vector<T> w(b.index_map(), b.bs());
                    w.set(0);
                    op(V[k], w);

                    // Modified Gram-Schmidt against previous basis vectors
                    for (int j = 0; j <= k; ++j) {
                        H[j][k] = inner_product(w, V[j]);
                        for (std::size_t i = 0; i < n; ++i)
                            w[i] -= H[j][k] * V[j][i];
                    }
                    H[k + 1][k] = std::sqrt(squared_norm(w));
                    for (std::size_t i = 0; i < n; ++i)
                        V[k + 1][i] = w[i] / H[k + 1][k];

                    // Apply previous Givens rotations
                    for (int j = 0; j < k; ++j) {
                        const T tmp = H[j][k];
                        H[j][k] = impl::conj_if_complex(c[j]) * tmp
                            + s[j] * H[j + 1][k];
                        H[j + 1][k]
                            = -s[j] * tmp + c[j] * H[j + 1][k];
                    }

                    // Compute the new Givens rotation
                    const T h_k = std::abs(H[k][k]);
                    const T h_k1 = std::abs(H[k + 1][k]);
                    if (h_k1 == 0) {
                        // The Krylov space is invariant: the system is
                        // already triangular. Keep `k` at the converged
                        // column for the back-substitution.
                        break;
                    }
                    const T denom = std::sqrt(h_k * h_k + h_k1 * h_k1);
                    c[k] = H[k][k] / denom;
                    s[k] = H[k + 1][k] / denom;
                    H[k][k] = denom;
                    H[k + 1][k] = 0;

                    gs[k + 1] = -s[k] * gs[k];
                    gs[k] = c[k] * gs[k];

                    if (std::abs(gs[k + 1]) <= tol) {
                        ++total_iter; // this column counts as an iteration
                        break;
                    }
                }

                // Back-substitute the upper triangular system to get y
                std::vector<T> y(k + 1, 0);
                for (int i = k; i >= 0; --i) {
                    y[i] = gs[i];
                    for (int j = i + 1; j <= k; ++j)
                        y[i] -= H[i][j] * y[j];
                    y[i] /= H[i][i];
                }

                // Update the solution: x += P (V y)
                Vector<T> tmp(b.index_map(), b.bs());
                tmp.set(0);
                for (int j = 0; j <= k; ++j)
                    for (std::size_t i = 0; i < n; ++i)
                        tmp[i] += V[j][i] * y[j];
                Vector<T> z(b.index_map(), b.bs());
                z.set(0);
                _apply_preconditioner(tmp, z);
                for (std::size_t i = 0; i < n; ++i)
                    x[i] += z[i];

                // Recompute the residual for the restart
                r = b;
                Vector<T> ax(b.index_map(), b.bs());
                ax.set(0);
                _A.mult(x, ax);
                for (std::size_t i = 0; i < n; ++i)
                    r[i] -= ax[i];
            }

            spdlog::warn("GMRES did not converge in {} iterations.",
                _max_iter);
            return _max_iter;
        }

        /// Right-preconditioned BiCGSTAB.
        int _bicgstab(Vector<T>& x, const Vector<T>& b) const
        {
            const std::size_t n = b.array().size();
            const auto tol = std::max(static_cast<decltype(squared_norm(b))>(_atol),
                static_cast<decltype(squared_norm(b))>(_rtol)
                    * std::sqrt(squared_norm(b)));

            // Residual r0 = b - A x0.
            Vector<T> r(b.index_map(), b.bs());
            r = b;
            if (_use_initial_guess) {
                Vector<T> ax(b.index_map(), b.bs());
                ax.set(0);
                _A.mult(x, ax);
                for (std::size_t i = 0; i < n; ++i)
                    r[i] -= ax[i];
            }
            if (std::sqrt(squared_norm(r)) <= tol)
                return 0;

            // Shadow residual rhat is fixed for the whole iteration.
            Vector<T> rhat(b.index_map(), b.bs());
            rhat = r;

            Vector<T> p(b.index_map(), b.bs());
            p.set(0);
            Vector<T> v(b.index_map(), b.bs());
            Vector<T> s(b.index_map(), b.bs());
            Vector<T> t(b.index_map(), b.bs());
            Vector<T> z(b.index_map(), b.bs());

            T rho_old {1};
            T alpha {1};
            T omega {1};

            for (int k = 0; k < _max_iter; ++k) {
                const T rho = inner_product(rhat, r);
                if (std::abs(static_cast<double>(rho)) == 0)
                    break; // breakdown: shadow residual orthogonal

                const T beta = (rho / rho_old) * (alpha / omega);
                // p = r + beta (p - omega v)
                for (std::size_t i = 0; i < n; ++i)
                    p[i] = r[i] + beta * (p[i] - omega * v[i]);

                // v = A P p
                v.set(0);
                _apply_preconditioner(p, z);
                {
                    Vector<T> ap(b.index_map(), b.bs());
                    ap.set(0);
                    _A.mult(z, ap);
                    v = ap;
                }

                const T rv = inner_product(rhat, v);
                if (std::abs(static_cast<double>(rv)) == 0)
                    break;
                alpha = rho / rv;

                // s = r - alpha v
                for (std::size_t i = 0; i < n; ++i)
                    s[i] = r[i] - alpha * v[i];

                if (std::sqrt(squared_norm(s)) <= tol) {
                    // x += alpha P p, done.
                    Vector<T> y(b.index_map(), b.bs());
                    y.set(0);
                    _apply_preconditioner(p, y);
                    for (std::size_t i = 0; i < n; ++i)
                        x[i] += alpha * y[i];
                    return k + 1;
                }

                // t = A P s
                t.set(0);
                _apply_preconditioner(s, z);
                {
                    Vector<T> ap(b.index_map(), b.bs());
                    ap.set(0);
                    _A.mult(z, ap);
                    t = ap;
                }

                const T ts = inner_product(t, s);
                const T tt = inner_product(t, t);
                if (std::abs(static_cast<double>(tt)) == 0)
                    break;
                omega = ts / tt;

                // x += alpha P p + omega P s
                Vector<T> y(b.index_map(), b.bs());
                y.set(0);
                _apply_preconditioner(p, z);
                for (std::size_t i = 0; i < n; ++i)
                    x[i] += alpha * z[i];
                y.set(0);
                _apply_preconditioner(s, z);
                for (std::size_t i = 0; i < n; ++i)
                    x[i] += omega * z[i];

                // r = s - omega t
                for (std::size_t i = 0; i < n; ++i)
                    r[i] = s[i] - omega * t[i];

                if (std::sqrt(squared_norm(r)) <= tol)
                    return k + 1;
                if (std::abs(static_cast<double>(omega)) == 0)
                    break; // breakdown

                rho_old = rho;
            }

            spdlog::warn("BiCGSTAB did not converge in {} iterations.", _max_iter);
            return _max_iter;
        }

        /// Apply the preconditioner to `x`, storing `y = P x`. If no
        /// preconditioner is set, `y = x`.
        void _apply_preconditioner(const Vector<T>& x, Vector<T>& y) const        {
            if (_P)
                _P->apply(x, y);
            else
                for (std::size_t i = 0; i < x.array().size(); ++i)
                    y[i] = x[i];
        }

        // The operator
        LinearOperator<T> _A;

        // Registered matrix (for preconditioner synthesis)
        std::shared_ptr<const MatrixCSR<T>> _matrix;

        // The preconditioner (none if null)
        std::shared_ptr<const Preconditioner<T>> _P;

        // Solver type
        SolverType _type = SolverType::cg;

        // Tolerances
        double _rtol = 1e-10;
        double _atol = 1e-14;
        int _max_iter = 1000;

        // Whether to use the initial guess
        bool _use_initial_guess = false;
    };

} // namespace hellofem::la
