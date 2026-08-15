// hellofem::la — BiCGSTAB iterative solver tests
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

    /// A shared index map for `n` scalar dofs.
    std::shared_ptr<common::IndexMap> imap(std::int32_t n)
    {
        return std::make_shared<common::IndexMap>(0, n);
    }

    /// Build a scalar MatrixCSR from a sparse entry list (row, col, value).
    la::MatrixCSR<double> make_matrix(std::int32_t n,
        const std::vector<std::tuple<std::int32_t, std::int32_t, double>>& triples)
    {
        la::SparsityPattern pattern(imap(n));
        for (auto [r, c, v] : triples)
            pattern.insert(r, c);
        pattern.finalize();

        la::MatrixCSR<double> A(pattern);
        for (std::int32_t r = 0; r < n; ++r) {
            std::vector<std::int32_t> row = {r};
            std::vector<std::int32_t> col_list;
            std::vector<double> val_list;
            for (auto [rr, cc, v] : triples)
                if (rr == r) {
                    col_list.push_back(cc);
                    val_list.push_back(v);
                }
            if (!col_list.empty())
                A.set(val_list, row, col_list);
        }
        return A;
    }

} // namespace

TEST_CASE("BiCGSTAB: solves a non-symmetric system", "[la]")
{
    // Same non-symmetric 3x3 system as the GMRES test.
    la::MatrixCSR<double> A = make_matrix(3,
        {{0, 0, 4.0}, {0, 1, 1.0}, {1, 0, 2.0}, {1, 1, 5.0}, {1, 2, 1.0},
            {2, 2, 6.0}});

    la::Vector<double> xstar(imap(3), 1);
    xstar.array()[0] = 1.0;
    xstar.array()[1] = 2.0;
    xstar.array()[2] = 3.0;
    la::Vector<double> b(imap(3), 1);
    b.set(0);
    A.mult(xstar, b);

    la::Vector<double> x(imap(3), 1);
    x.set(0);
    la::KrylovSolver<double> solver;
    solver.set_operator(A.as_operator());
    solver.set_solver_type("bicgstab");
    solver.set_tolerances(1e-10, 1e-14, 200);
    int it = solver.solve(x, b);
    REQUIRE(it > 0);
    for (std::int32_t i = 0; i < 3; ++i)
        REQUIRE(std::abs(x.array()[i] - xstar.array()[i]) < 1e-8);
}

TEST_CASE("BiCGSTAB: agrees with GMRES on a nonsymmetric 5x5", "[la]")
{
    // Slightly larger non-symmetric system.
    la::MatrixCSR<double> A = make_matrix(5,
        {{0, 0, 6.0}, {0, 1, 1.0}, {0, 2, 2.0},
            {1, 0, 1.0}, {1, 1, 7.0}, {1, 3, 1.0},
            {2, 0, 2.0}, {2, 2, 8.0}, {2, 4, 1.0},
            {3, 1, 1.0}, {3, 3, 9.0},
            {4, 2, 1.0}, {4, 4, 10.0}});

    la::Vector<double> xstar(imap(5), 1);
    for (std::int32_t i = 0; i < 5; ++i)
        xstar.array()[i] = std::sin(static_cast<double>(i + 1));
    la::Vector<double> b(imap(5), 1);
    b.set(0);
    A.mult(xstar, b);

    // BiCGSTAB
    la::Vector<double> xb(imap(5), 1);
    xb.set(0);
    la::KrylovSolver<double> sb;
    sb.set_operator(A.as_operator());
    sb.set_solver_type("bicgstab");
    sb.set_tolerances(1e-12, 1e-14, 200);
    REQUIRE(sb.solve(xb, b) > 0);

    // GMRES
    la::Vector<double> xg(imap(5), 1);
    xg.set(0);
    la::KrylovSolver<double> sg;
    sg.set_operator(A.as_operator());
    sg.set_solver_type("gmres");
    sg.set_tolerances(1e-12, 1e-14, 200);
    REQUIRE(sg.solve(xg, b) > 0);

    for (std::int32_t i = 0; i < 5; ++i)
        REQUIRE(std::abs(xb.array()[i] - xstar.array()[i]) < 1e-9);
}

TEST_CASE("BiCGSTAB: solves a symmetric positive-definite (Poisson) system", "[la]")
{
    // 1D Poisson tridiagonal: diag 2, off -1.
    const std::int32_t n = 8;
    la::MatrixCSR<double> A = make_matrix(n, [&] {
        std::vector<std::tuple<std::int32_t, std::int32_t, double>> t;
        for (std::int32_t i = 0; i < n; ++i) {
            t.emplace_back(i, i, 2.0);
            if (i > 0)
                t.emplace_back(i, i - 1, -1.0);
            if (i < n - 1)
                t.emplace_back(i, i + 1, -1.0);
        }
        return t;
    }());

    la::Vector<double> xstar(imap(n), 1);
    for (std::int32_t i = 0; i < n; ++i)
        xstar.array()[i] = static_cast<double>(i + 1);
    la::Vector<double> b(imap(n), 1);
    b.set(0);
    A.mult(xstar, b);

    la::Vector<double> x(imap(n), 1);
    x.set(0);
    la::KrylovSolver<double> solver;
    solver.set_operator(A.as_operator());
    solver.set_solver_type("bicgstab");
    solver.set_tolerances(1e-10, 1e-14, 200);
    REQUIRE(solver.solve(x, b) > 0);
    for (std::int32_t i = 0; i < n; ++i)
        REQUIRE(std::abs(x.array()[i] - xstar.array()[i]) < 1e-8);
}
