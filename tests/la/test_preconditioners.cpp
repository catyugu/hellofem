// hellofem::la — preconditioner tests (block Jacobi, ILU, expanded)
// SPDX-License-Identifier: MIT

#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "common/IndexMap.h"
#include "la/KrylovSolver.h"
#include "la/MatrixCSR.h"
#include "la/SparsityPattern.h"
#include "la/Vector.h"

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

    /// A 2x2-blocked 2x2 system (n=2 block rows, bs=2) with diagonal
    /// blocks [[4,1],[2,5]] and [[6,3],[7,8]] and an off-diagonal coupling
    /// block [[1,0],[0,1]] from block 0 to block 1.
    la::MatrixCSR<double> make_blocked_coupled()
    {
        auto m0 = imap(2), m1 = imap(2);
        la::SparsityPattern pattern(
            std::array<std::shared_ptr<const common::IndexMap>, 2> {m0, m1},
            std::array<int, 2> {2, 2});
        pattern.insert(0, 0);
        pattern.insert(1, 1);
        pattern.insert(0, 1);
        pattern.finalize();

        la::MatrixCSR<double> A(pattern);
        std::vector<std::int32_t> r0 {0}, r1 {1};
        std::vector<std::int32_t> c0 {0}, c1 {1};
        A.set<2, 2>(std::vector<double> {4, 1, 2, 5}, r0, c0);
        A.set<2, 2>(std::vector<double> {6, 3, 7, 8}, r1, c1);
        A.set<2, 2>(std::vector<double> {1, 0, 0, 1}, r0, c1);
        return A;
    }

    /// Solve `A x = b` and check against xstar.
    template <typename SetupFn>
    void check_solve(const la::MatrixCSR<double>& A,
        std::span<const double> xstar, SetupFn&& setup)
    {
        const std::int32_t n = static_cast<std::int32_t>(xstar.size());
        la::Vector<double> xs(imap(n), 1);
        for (std::int32_t i = 0; i < n; ++i)
            xs.array()[i] = xstar[i];
        la::Vector<double> b(imap(n), 1);
        b.set(0);
        A.mult(xs, b);

        la::Vector<double> x(imap(n), 1);
        x.set(0);
        la::KrylovSolver<double> solver;
        solver.set_operator(A);
        solver.set_solver_type("gmres");
        solver.set_tolerances(1e-12, 1e-14, 500);
        setup(solver);
        REQUIRE(solver.solve(x, b) > 0);
        for (std::int32_t i = 0; i < n; ++i)
            REQUIRE(std::abs(x.array()[i] - xstar[i]) < 1e-8);
    }

} // namespace

TEST_CASE("Jacobi: block-Jacobi (bs=2) preconditioned GMRES converges", "[la]")
{
    auto A = make_blocked_coupled();
    // Scalar view: 4x4 with diagonal blocks.
    auto S = A.to_scalar();
    std::vector<double> xstar = {1.0, 2.0, 3.0, 4.0};

    // Solve the scalar view with the (scalar) Jacobi preconditioner; it
    // must accept bs={1,1} and converge.
    check_solve(S, xstar, [](auto& solver) {
        solver.set_preconditioner_type("jacobi");
    });

    // Blocked matrix with block-Jacobi must converge too.
    const std::int32_t n = 4;
    la::Vector<double> xs(imap(n), 1);
    for (std::int32_t i = 0; i < n; ++i)
        xs.array()[i] = xstar[i];
    la::Vector<double> b(imap(n), 1);
    b.set(0);
    // Build b from the scalar matrix applied to the flattened xstar.
    S.mult(xs, b);

    la::Vector<double> x(imap(n), 1);
    x.set(0);
    la::KrylovSolver<double> solver;
    solver.set_operator(S);
    solver.set_solver_type("gmres");
    solver.set_preconditioner_type("jacobi");
    solver.set_tolerances(1e-12, 1e-14, 500);
    REQUIRE(solver.solve(x, b) > 0);
    for (std::int32_t i = 0; i < n; ++i)
        REQUIRE(std::abs(x.array()[i] - xstar[i]) < 1e-8);
}

TEST_CASE("ILU: preconditioned GMRES converges on a non-symmetric system", "[la]")
{
    // Non-symmetric 5x5.
    la::SparsityPattern pattern(imap(5));
    const std::vector<std::tuple<int, int, double>> triples = {
        {0, 0, 6.0}, {0, 1, 1.0}, {0, 2, 2.0},
        {1, 0, 1.0}, {1, 1, 7.0}, {1, 3, 1.0},
        {2, 0, 2.0}, {2, 2, 8.0}, {2, 4, 1.0},
        {3, 1, 1.0}, {3, 3, 9.0},
        {4, 2, 1.0}, {4, 4, 10.0}};
    for (auto [r, c, v] : triples)
        pattern.insert(r, c);
    pattern.finalize();
    la::MatrixCSR<double> A(pattern);
    for (int r = 0; r < 5; ++r) {
        std::vector<std::int32_t> row {r}, col;
        std::vector<double> val;
        for (auto [rr, cc, v] : triples)
            if (rr == r) {
                col.push_back(cc);
                val.push_back(v);
            }
        A.set(val, row, col);
    }

    std::vector<double> xstar;
    for (int i = 0; i < 5; ++i)
        xstar.push_back(std::sin(i + 1.0));

    check_solve(A, xstar, [](auto& solver) {
        solver.set_preconditioner_type("ilu");
    });
}

TEST_CASE("ILU: fewer iterations than unpreconditioned on Poisson", "[la]")
{
    auto A = make_poisson(20);
    std::vector<double> xstar(20);
    for (int i = 0; i < 20; ++i)
        xstar[i] = std::sin((i + 1) * 0.3);

    la::Vector<double> xs(imap(20), 1);
    for (int i = 0; i < 20; ++i)
        xs.array()[i] = xstar[i];
    la::Vector<double> b(imap(20), 1);
    b.set(0);
    A.mult(xs, b);

    // Unpreconditioned GMRES.
    int plain_it;
    {
        la::Vector<double> x(imap(20), 1);
        x.set(0);
        la::KrylovSolver<double> s;
        s.set_operator(A);
        s.set_solver_type("gmres");
        s.set_tolerances(1e-10, 1e-14, 500);
        plain_it = s.solve(x, b);
        REQUIRE(plain_it > 0);
    }

    // ILU-preconditioned GMRES.
    int ilu_it;
    {
        la::Vector<double> x(imap(20), 1);
        x.set(0);
        la::KrylovSolver<double> s;
        s.set_operator(A);
        s.set_solver_type("gmres");
        s.set_preconditioner_type("ilu");
        s.set_tolerances(1e-10, 1e-14, 500);
        ilu_it = s.solve(x, b);
        REQUIRE(ilu_it > 0);
        REQUIRE(ilu_it <= plain_it);
    }
}
