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
        // COMSOL's Anderson settings, with a full mixing parameter: a full
        // step is what makes an operator that does not read the solution reach
        // its fixed point in one iteration, where the damped step of a smaller
        // value approaches it geometrically at the cost of a re-assembly per
        // iteration.
        nls::AndersonConfig cfg;
        cfg.mixing = 1.0;
        cfg.linear = settings;

        auto result = nls::anderson_picard<double>(
            [&](const la::Vector<double>&) {
                la::MatrixCSR<double> A(pattern);
                la::Vector<double> b(x.index_map(), x.bs());
                assemble(A, b);
                return std::make_pair(std::move(A), std::move(b));
            },
            x, solver, cfg);
        if (not result.converged)
            throw std::runtime_error(
                "solve_system: the nonlinear iteration did not converge");
        spdlog::debug("nonlinear: {} iterations, {} Krylov iterations",
            result.iterations, result.krylov_iterations);
        return result.iterations;
    }

} // namespace hellofem::app
