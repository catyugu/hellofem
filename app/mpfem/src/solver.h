// hellofem::app — the solve of a field's system
// SPDX-License-Identifier: MIT
#pragma once

#include "la/LinearSolver.h"
#include "la/MatrixCSR.h"
#include "la/SparsityPattern.h"
#include "la/Vector.h"

#include <functional>

namespace hellofem::app {

    /// Whether a Krylov solve that used `iterations` of `max_iterations`
    /// solved its system.
    ///
    /// A solver that stops at its iteration cap has not: it returns the cap
    /// and leaves an intermediate iterate in `x`. Reading that iterate as the
    /// field's value reports a non-solution as a result — a field that is
    /// silently wrong, in a run that looks like it succeeded.
    bool converged(int iterations, int max_iterations);

    /// Defaults of the fixed-point iteration that solves a field's system
    /// (Anderson-accelerated Picard).
    struct NonlinearSettings {
        int depth = 5;
        int warmup_iterations = 3;
        /// Anderson damping. A full step (1.0) is what makes an operator that
        /// does not read the solution reach its fixed point in one iteration:
        /// the damped step of a smaller value approaches it geometrically, at
        /// the cost of a re-assembly per iteration.
        double dampening = 1.0;
        double max_growth = 1.5;
        double relative_tolerance = 1e-8;
        double absolute_tolerance = 1e-12;
        int max_iterations = 50;
        /// Seed the inner linear solve with the current iterate.
        bool warm_start = true;
    };

    /// Solve the system `assemble` builds for the unknown `x`, starting from
    /// `x` as the initial guess.
    ///
    /// One iteration solves both a nonlinear law and a linear one: the
    /// Anderson-accelerated Picard iteration converges in its first step where
    /// the operator does not read the solution, and iterates it where it does.
    /// `assemble` must therefore refresh the material state from the current
    /// `x` on every call.
    ///
    /// `solver` belongs to the field and is kept across calls, so a level
    /// whose operator is the one of the level before pays for its
    /// factorization once (see `la::LinearSolver`).
    /// @return Iterations used.
    int solve_system(
        const std::function<void(la::MatrixCSR<double>&, la::Vector<double>&)>& assemble,
        la::Vector<double>& x, const la::SparsityPattern& pattern,
        la::LinearSolver<double>& solver, const la::LinearSettings& settings);

} // namespace hellofem::app
