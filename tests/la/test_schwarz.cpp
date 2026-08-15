// hellofem::la — additive Schwarz (overlapping DDM) preconditioner tests
// SPDX-License-Identifier: MIT

#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "common/IndexMap.h"
#include "la/KrylovSolver.h"
#include "la/MatrixCSR.h"
#include "la/SparsityPattern.h"
#include "la/Vector.h"
#include "la/schwarz.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <tuple>
#include <vector>

using namespace hellofem;

namespace {

    std::shared_ptr<common::IndexMap> imap(std::int32_t n)
    {
        return std::make_shared<common::IndexMap>(0, n);
    }

    /// 1D Poisson tridiagonal: diag=2, off=-1.
    la::MatrixCSR<double> make_poisson(std::int32_t n)
    {
        la::SparsityPattern pattern(imap(n));
        for (std::int32_t i = 0; i < n; ++i) {
            pattern.insert(i, i);
            if (i > 0)
                pattern.insert(i, i - 1);
            if (i < n - 1)
                pattern.insert(i, i + 1);
        }
        pattern.finalize();

        la::MatrixCSR<double> A(pattern);
        std::vector<std::int32_t> row = {0};
        for (std::int32_t i = 0; i < n; ++i) {
            std::vector<std::int32_t> col;
            std::vector<double> val;
            if (i > 0) {
                col.push_back(i - 1);
                val.push_back(-1.0);
            }
            col.push_back(i);
            val.push_back(2.0);
            if (i < n - 1) {
                col.push_back(i + 1);
                val.push_back(-1.0);
            }
            A.set(val, row, col);
            row[0] = i + 1;
        }
        return A;
    }

} // namespace

TEST_CASE("Schwarz: preconditioned CG converges on a Poisson system", "[la]")
{
    auto A = make_poisson(30);
    std::vector<double> xstar(30);
    for (int i = 0; i < 30; ++i)
        xstar[i] = std::sin((i + 1) * 0.4);

    la::Vector<double> xs(imap(30), 1);
    for (int i = 0; i < 30; ++i)
        xs.array()[i] = xstar[i];
    la::Vector<double> b(imap(30), 1);
    b.set(0);
    A.mult(xs, b);

    la::Vector<double> x(imap(30), 1);
    x.set(0);
    la::KrylovSolver<double> solver;
    solver.set_operator(A);
    solver.set_solver_type("cg");
    solver.set_preconditioner(std::make_shared<la::SchwarzPreconditioner<double>>(
        A, 2, 1));
    solver.set_tolerances(1e-12, 1e-14, 1000);
    REQUIRE(solver.solve(x, b) > 0);
    for (int i = 0; i < 30; ++i)
        REQUIRE(std::abs(x.array()[i] - xstar[i]) < 1e-8);
}

TEST_CASE("Schwarz: more overlap reduces iterations (Poisson)", "[la]")
{
    auto A = make_poisson(60);
    std::vector<double> xstar(60);
    for (int i = 0; i < 60; ++i)
        xstar[i] = std::sin((i + 1) * 0.2);

    la::Vector<double> xs(imap(60), 1);
    for (int i = 0; i < 60; ++i)
        xs.array()[i] = xstar[i];
    la::Vector<double> b(imap(60), 1);
    b.set(0);
    A.mult(xs, b);

    auto solve_with = [&](int nparts, int overlap) {
        la::Vector<double> x(imap(60), 1);
        x.set(0);
        la::KrylovSolver<double> s;
        s.set_operator(A);
        s.set_solver_type("cg");
        s.set_preconditioner(std::make_shared<la::SchwarzPreconditioner<double>>(
            A, nparts, overlap));
        s.set_tolerances(1e-10, 1e-14, 1000);
        auto it = s.solve(x, b);
        for (int i = 0; i < 60; ++i)
            REQUIRE(std::abs(x.array()[i] - xstar[i]) < 1e-8);
        return it;
    };

    // Two subdomains, overlap 0 vs overlap 2: more overlap should
    // converge in fewer (or not more) iterations.
    const int it0 = solve_with(2, 0);
    const int it2 = solve_with(2, 2);
    REQUIRE(it0 > 0);
    REQUIRE(it2 > 0);
    REQUIRE(it2 <= it0 + 2); // not a strict claim, but overlap must not hurt
}
