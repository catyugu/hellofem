// hellofem::app — the solve of a field's system
// SPDX-License-Identifier: MIT
#pragma once

#include "la/LinearSolver.h"
#include "la/MatrixCSR.h"
#include "la/SparsityPattern.h"
#include "la/Vector.h"

#include <functional>

namespace hellofem::app {

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
