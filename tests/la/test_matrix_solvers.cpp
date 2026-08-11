// hellofem::la — Vector, MatrixCSR and iterative solver tests
// SPDX-License-Identifier: MIT

#include "la/KrylovSolver.h"
#include "la/MatrixCSR.h"
#include "la/Vector.h"

#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "common/IndexMap.h"
#include "la/SparsityPattern.h"

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

    /// 1D Poisson stiffness-like tridiagonal matrix: diag=2, off=-1.
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

TEST_CASE("Vector: set, array, norms and inner product", "[la]")
{
    auto map = imap(3);
    la::Vector<double> v(map, 1);
    v.set(0);
    REQUIRE(v.array().size() == 3);
    REQUIRE(v.bs() == 1);

    v.array()[0] = 1.0;
    v.array()[1] = 2.0;
    v.array()[2] = 3.0;

    REQUIRE(la::squared_norm(v) == Catch::Approx(14.0));
    REQUIRE(la::norm(v, la::Norm::l2) == Catch::Approx(std::sqrt(14.0)));
    REQUIRE(la::norm(v, la::Norm::l1) == Catch::Approx(6.0));
    REQUIRE(la::norm(v, la::Norm::linf) == Catch::Approx(3.0));

    la::Vector<double> w(map, 1);
    w.array()[0] = 2.0;
    w.array()[1] = 3.0;
    w.array()[2] = 4.0;
    REQUIRE(la::inner_product(v, w) == Catch::Approx(20.0));
}

TEST_CASE("MatrixCSR: build, set block, to_dense and mult", "[la]")
{
    // 3x3 matrix:
    // [4 1 0]
    // [2 5 0]
    // [0 0 6]
    const std::vector<std::pair<std::int32_t, std::int32_t>> entries
        = {{0, 0}, {0, 1}, {1, 0}, {1, 1}, {2, 2}};

    la::SparsityPattern pattern(imap(3));
    for (auto [r, c] : entries)
        pattern.insert(r, c);
    pattern.finalize();

    la::MatrixCSR<double> A(pattern);

    // Set the 2x2 block at rows {0,1} x cols {0,1}, then (2,2).
    std::vector<std::int32_t> rows01 = {0, 1};
    std::vector<std::int32_t> cols01 = {0, 1};
    std::vector<double> block01 = {4.0, 1.0, 2.0, 5.0};
    A.set(block01, rows01, cols01);

    std::vector<std::int32_t> row2 = {2};
    std::vector<std::int32_t> col2 = {2};
    std::vector<double> val2 = {6.0};
    A.set(val2, row2, col2);

    const auto dense = A.to_dense();
    REQUIRE(dense.size() == 9);
    std::vector<double> expected = {4, 1, 0, 2, 5, 0, 0, 0, 6};
    for (std::size_t i = 0; i < 9; ++i)
        REQUIRE(dense[i] == Catch::Approx(expected[i]));

    // mult: y = A x
    la::Vector<double> x(imap(3), 1);
    x.array()[0] = 1.0;
    x.array()[1] = 2.0;
    x.array()[2] = 3.0;
    la::Vector<double> y(imap(3), 1);
    y.set(0);
    A.mult(x, y);
    REQUIRE(y.array()[0] == Catch::Approx(4 * 1 + 1 * 2 + 0 * 3));
    REQUIRE(y.array()[1] == Catch::Approx(2 * 1 + 5 * 2 + 0 * 3));
    REQUIRE(y.array()[2] == Catch::Approx(0 * 1 + 0 * 2 + 6 * 3));

    // add accumulates: A += block again
    A.add(block01, rows01, cols01);
    const auto dense2 = A.to_dense();
    REQUIRE(dense2[0] == Catch::Approx(8.0));
    REQUIRE(dense2[4] == Catch::Approx(10.0));
}

TEST_CASE("CG: solves a small SPD system", "[la]")
{
    const std::int32_t n = 4;
    la::MatrixCSR<double> A = make_poisson(n);

    // Known solution and rhs.
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
    solver.set_solver_type("cg");
    solver.set_tolerances(1e-10, 1e-14, 200);
    int it = solver.solve(x, b);
    REQUIRE(it > 0);
    for (std::int32_t i = 0; i < n; ++i)
        REQUIRE(std::abs(x.array()[i] - xstar.array()[i]) < 1e-8);
}

TEST_CASE("Jacobi and AMG preconditioned CG converge on a Poisson system", "[la]")
{
    const std::int32_t n = 10;
    la::MatrixCSR<double> A = make_poisson(n);

    // rhs = A * xstar with a smooth solution.
    std::vector<double> xstar;
    for (std::int32_t i = 0; i < n; ++i)
        xstar.push_back(std::sin((i + 1) * 0.5));

    auto xstar_v = la::Vector<double>(imap(n), 1);
    for (std::int32_t i = 0; i < n; ++i)
        xstar_v.array()[i] = xstar[i];
    la::Vector<double> b(imap(n), 1);
    b.set(0);
    A.mult(xstar_v, b);

    // Plain CG
    int plain_it;
    {
        la::Vector<double> x(imap(n), 1);
        x.set(0);
        la::KrylovSolver<double> solver;
        solver.set_operator(A.as_operator());
        solver.set_solver_type("cg");
        solver.set_tolerances(1e-10, 1e-14, 200);
        plain_it = solver.solve(x, b);
        REQUIRE(plain_it > 0);
        for (std::int32_t i = 0; i < n; ++i)
            REQUIRE(std::abs(x.array()[i] - xstar[i]) < 1e-8);
    }

    // Jacobi-preconditioned CG
    {
        la::Vector<double> x(imap(n), 1);
        x.set(0);
        la::KrylovSolver<double> solver;
        solver.set_operator(A);
        solver.set_solver_type("cg");
        solver.set_preconditioner_type("jacobi");
        solver.set_tolerances(1e-10, 1e-14, 200);
        int it = solver.solve(x, b);
        REQUIRE(it > 0);
        for (std::int32_t i = 0; i < n; ++i)
            REQUIRE(std::abs(x.array()[i] - xstar[i]) < 1e-8);
    }

    // AMG-preconditioned CG, which must converge in fewer iterations than
    // plain CG (verifies the amgcl preconditioner is actually applied).
    {
        la::Vector<double> x(imap(n), 1);
        x.set(0);
        la::KrylovSolver<double> solver;
        solver.set_operator(A);
        solver.set_solver_type("cg");
        solver.set_preconditioner_type("amg");
        solver.set_tolerances(1e-10, 1e-14, 200);
        int amg_it = solver.solve(x, b);
        REQUIRE(amg_it > 0);
        REQUIRE(amg_it < plain_it);
        for (std::int32_t i = 0; i < n; ++i)
            REQUIRE(std::abs(x.array()[i] - xstar[i]) < 1e-8);
    }
}

TEST_CASE("GMRES: solves a non-symmetric system", "[la]")
{
    // Non-symmetric 3x3 system.
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
    solver.set_solver_type("gmres");
    solver.set_tolerances(1e-10, 1e-14, 200);
    int it = solver.solve(x, b);
    REQUIRE(it > 0);
    for (std::int32_t i = 0; i < 3; ++i)
        REQUIRE(std::abs(x.array()[i] - xstar.array()[i]) < 1e-8);
}

TEST_CASE("MatrixCSR: expanded block mode throws", "[la]")
{
    la::SparsityPattern pattern(imap(2));
    std::vector<std::int32_t> diag = {0, 1};
    pattern.insert_diagonal(diag);
    pattern.finalize();
    la::MatrixCSR<double> A(pattern);
    REQUIRE_NOTHROW(A); // compact mode is fine
    REQUIRE_THROWS(la::MatrixCSR<double>(pattern, la::BlockMode::expanded));
}
