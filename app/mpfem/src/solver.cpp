// hellofem::app — solver drivers (linear Krylov defaults, nonlinear iteration)
// SPDX-License-Identifier: MIT

#include "solver.h"

#include "la/KrylovSolver.h"
#include "la/direct.h"
#include "nls/AndersonPicard.h"

#include <spdlog/spdlog.h>

#include <stdexcept>
#include <utility>

namespace hellofem::app {
    namespace {

        /// Whether a solver type names a direct factorization. A direct solve
        /// is what an operator a factorization suits better asks for; the la
        /// layer owns which factorization that is.
        bool is_direct(const std::string& type) { return type == "direct"; }

    } // namespace

    bool converged(int iterations, int max_iterations)
    {
        return iterations < max_iterations;
    }

    void solve_linear(const la::MatrixCSR<double>& A, la::Vector<double>& x,
        const la::Vector<double>& b, bool warm_start,
        const LinearSettings& settings)
    {
        if (is_direct(settings.solver_type)) {
            la::DirectSolver solver(A);
            solver.solve(x, b);
            return;
        }
        la::KrylovSolver<double> solver;
        solver.set_operator(A);
        solver.set_solver_type(settings.solver_type);
        solver.set_preconditioner_type(settings.preconditioner_type);
        solver.set_tolerances(
            settings.rtol, settings.atol, settings.max_iterations);
        solver.set_initial_guess(warm_start);
        const int iterations = solver.solve(x, b);
        if (not converged(iterations, settings.max_iterations))
            throw std::runtime_error("solve_linear: " + settings.solver_type
                + " with " + settings.preconditioner_type + " preconditioning did not "
                + "converge in " + std::to_string(settings.max_iterations)
                + " iterations at rtol " + std::to_string(settings.rtol));
        spdlog::debug("linear: {} iterations", iterations);
    }

    int solve_system(
        const std::function<void(la::MatrixCSR<double>&, la::Vector<double>&)>& assemble,
        la::Vector<double>& x, const la::SparsityPattern& pattern, bool nonlinear,
        bool warm_start, const LinearSettings& settings)
    {
        if (not nonlinear) {
            la::MatrixCSR<double> A(pattern);
            la::Vector<double> b(x.index_map(), x.bs());
            assemble(A, b);
            solve_linear(A, x, b, warm_start, settings);
            return 0;
        }

        // The fixed-point iteration solves its frozen system with a Krylov
        // method; a direct solve is not wired into it.
        if (is_direct(settings.solver_type))
            throw std::runtime_error("solve_system: a direct solve of a "
                                     "nonlinear system is not supported.");

        const NonlinearSettings cfg;
        nls::AndersonConfig picard;
        picard.depth = cfg.depth;
        picard.warmup_iters = cfg.warmup_iterations;
        picard.dampening = cfg.dampening;
        picard.max_growth = cfg.max_growth;
        picard.relative_tolerance = cfg.relative_tolerance;
        picard.absolute_tolerance = cfg.absolute_tolerance;
        picard.max_iterations = cfg.max_iterations;
        picard.linear_solver_type = settings.solver_type;
        picard.preconditioner_type = settings.preconditioner_type;
        picard.krylov_rtol = settings.rtol;
        picard.krylov_atol = settings.atol;
        picard.krylov_max_iter = settings.max_iterations;
        picard.warm_start_linear_solve = cfg.warm_start;

        auto result = nls::anderson_picard<double>(
            [&](const la::Vector<double>&) {
                la::MatrixCSR<double> A(pattern);
                la::Vector<double> b(x.index_map(), x.bs());
                assemble(A, b);
                return std::make_pair(std::move(A), std::move(b));
            },
            x, picard);
        if (not result.converged)
            throw std::runtime_error(
                "solve_system: the nonlinear iteration did not converge");
        spdlog::debug("nonlinear: {} iterations, {} Krylov iterations",
            result.iterations, result.krylov_iterations);
        return result.iterations;
    }

} // namespace hellofem::app
