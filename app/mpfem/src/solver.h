// hellofem::app — solver drivers (linear Krylov defaults, nonlinear iteration)
// SPDX-License-Identifier: MIT
#pragma once

#include "la/MatrixCSR.h"
#include "la/SparsityPattern.h"
#include "la/Vector.h"

#include <functional>

namespace hellofem::app {

    /// Defaults of the app's Krylov solves: conjugate gradients with
    /// algebraic multigrid preconditioning, at a tolerance that makes the
    /// linear error negligible against the discretization error.
    struct LinearSettings {
        std::string solver_type = "cg";
        std::string preconditioner_type = "amg";
        double rtol = 1e-12;
        double atol = 1e-14;
        int max_iterations = 2000;
    };

    /// Whether a Krylov solve that used `iterations` of `max_iterations`
    /// solved its system.
    ///
    /// A solver that stops at its iteration cap has not: it returns the cap
    /// and leaves an intermediate iterate in `x`. Reading that iterate as the
    /// field's value reports a non-solution as a result — a field that is
    /// silently wrong, in a run that looks like it succeeded.
    bool converged(int iterations, int max_iterations);

    /// Solve `A x = b`, taking the current `x` as the initial guess when
    /// `warm_start` is set (a previous time level or the previous
    /// linearization is a good starting point). Throws when the solve does
    /// not converge within the iteration cap.
    void solve_linear(const la::MatrixCSR<double>& A, la::Vector<double>& x,
        const la::Vector<double>& b, bool warm_start = false);

    /// Defaults of the fixed-point iteration used for nonlinear material
    /// laws (Anderson-accelerated Picard).
    struct NonlinearSettings {
        int depth = 5;
        int warmup_iterations = 3;
        double dampening = 0.8;
        double max_growth = 1.5;
        double relative_tolerance = 1e-8;
        double absolute_tolerance = 1e-12;
        int max_iterations = 50;
        /// Seed the inner linear solve with the current iterate.
        bool warm_start = true;
    };

    /// Solve the system assembled by `assemble` for the unknown `x`,
    /// starting from `x` as the initial guess. A linear problem needs a
    /// single solve; a nonlinear one is iterated to convergence, and
    /// `assemble` must refresh the material state from the current `x` on
    /// every call.
    /// @return Iterations used (0 for the single solve of a linear problem).
    int solve_system(
        const std::function<void(la::MatrixCSR<double>&, la::Vector<double>&)>& assemble,
        la::Vector<double>& x, const la::SparsityPattern& pattern, bool nonlinear,
        bool warm_start = false);

} // namespace hellofem::app
