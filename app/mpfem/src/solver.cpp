// hellofem::app — solver drivers (linear Krylov defaults, nonlinear iteration)
// SPDX-License-Identifier: MIT

#include "solver.h"

#include "nls/AndersonPicard.h"

#include <spdlog/spdlog.h>

#include <stdexcept>
#include <utility>

namespace hellofem::app {

    bool converged(int iterations, int max_iterations)
    {
        return iterations < max_iterations;
    }

    int solve_system(
        const std::function<void(la::MatrixCSR<double>&, la::Vector<double>&)>& assemble,
        la::Vector<double>& x, const la::SparsityPattern& pattern,
        la::LinearSolver<double>& solver, const la::LinearSettings& settings)
    {
        const NonlinearSettings cfg;
        nls::AndersonConfig picard;
        picard.depth = cfg.depth;
        picard.warmup_iters = cfg.warmup_iterations;
        picard.dampening = cfg.dampening;
        picard.max_growth = cfg.max_growth;
        picard.relative_tolerance = cfg.relative_tolerance;
        picard.absolute_tolerance = cfg.absolute_tolerance;
        picard.max_iterations = cfg.max_iterations;
        picard.linear = settings;
        picard.warm_start_linear_solve = cfg.warm_start;

        auto result = nls::anderson_picard<double>(
            [&](const la::Vector<double>&) {
                la::MatrixCSR<double> A(pattern);
                la::Vector<double> b(x.index_map(), x.bs());
                assemble(A, b);
                return std::make_pair(std::move(A), std::move(b));
            },
            x, solver, picard);
        if (not result.converged)
            throw std::runtime_error(
                "solve_system: the nonlinear iteration did not converge");
        spdlog::debug("nonlinear: {} iterations, {} Krylov iterations",
            result.iterations, result.krylov_iterations);
        return result.iterations;
    }

} // namespace hellofem::app
