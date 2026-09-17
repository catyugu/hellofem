// hellofem::app — solver drivers (linear Krylov defaults, nonlinear iteration)
// SPDX-License-Identifier: MIT

#include "solver.h"

#include "la/KrylovSolver.h"
#include "nls/AndersonPicard.h"

#include <spdlog/spdlog.h>

#include <stdexcept>
#include <utility>

namespace hellofem::app {

    void solve_linear(const la::MatrixCSR<double>& A, la::Vector<double>& x,
        const la::Vector<double>& b, bool warm_start)
    {
        LinearSettings cfg;
        la::KrylovSolver<double> solver;
        solver.set_operator(A);
        solver.set_solver_type(cfg.solver_type);
        solver.set_preconditioner_type(cfg.preconditioner_type);
        solver.set_tolerances(cfg.rtol, cfg.atol, cfg.max_iterations);
        solver.set_initial_guess(warm_start);
        const int iterations = solver.solve(x, b);
        spdlog::debug("linear: {} iterations", iterations);
    }

    int solve_system(
        const std::function<void(la::MatrixCSR<double>&, la::Vector<double>&)>& assemble,
        la::Vector<double>& x, const la::SparsityPattern& pattern, bool nonlinear,
        bool warm_start)
    {
        if (not nonlinear) {
            la::MatrixCSR<double> A(pattern);
            la::Vector<double> b(x.index_map(), x.bs());
            assemble(A, b);
            solve_linear(A, x, b, warm_start);
            return 0;
        }

        const LinearSettings linear;
        const NonlinearSettings cfg;
        nls::AndersonConfig picard;
        picard.depth = cfg.depth;
        picard.warmup_iters = cfg.warmup_iterations;
        picard.dampening = cfg.dampening;
        picard.max_growth = cfg.max_growth;
        picard.relative_tolerance = cfg.relative_tolerance;
        picard.absolute_tolerance = cfg.absolute_tolerance;
        picard.max_iterations = cfg.max_iterations;
        picard.linear_solver_type = linear.solver_type;
        picard.preconditioner_type = linear.preconditioner_type;
        picard.krylov_rtol = linear.rtol;
        picard.krylov_atol = linear.atol;
        picard.krylov_max_iter = linear.max_iterations;
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
