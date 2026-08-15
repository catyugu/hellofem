// hellofem::la — parallel sparse matrix-vector product tests
// SPDX-License-Identifier: MIT

#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "common/IndexMap.h"
#include "la/MatrixCSR.h"
#include "la/SparsityPattern.h"
#include "la/Vector.h"

#include <tbb/global_control.h>

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

    /// 2D Poisson-like 5-point stencil on an n x n grid (n*n dofs).
    la::MatrixCSR<double> make_2d_poisson(std::int32_t n)
    {
        const std::int32_t N = n * n;
        la::SparsityPattern pattern(imap(N));
        for (std::int32_t i = 0; i < n; ++i)
            for (std::int32_t j = 0; j < n; ++j) {
                const std::int32_t r = i * n + j;
                pattern.insert(r, r);
                if (i > 0)
                    pattern.insert(r, r - n);
                if (i < n - 1)
                    pattern.insert(r, r + n);
                if (j > 0)
                    pattern.insert(r, r - 1);
                if (j < n - 1)
                    pattern.insert(r, r + 1);
            }
        pattern.finalize();

        la::MatrixCSR<double> A(pattern);
        std::vector<std::int32_t> row {0};
        for (std::int32_t i = 0; i < n; ++i)
            for (std::int32_t j = 0; j < n; ++j) {
                const std::int32_t r = i * n + j;
                std::vector<std::int32_t> col;
                std::vector<double> val;
                if (i > 0) {
                    col.push_back(r - n);
                    val.push_back(-1.0);
                }
                if (i < n - 1) {
                    col.push_back(r + n);
                    val.push_back(-1.0);
                }
                if (j > 0) {
                    col.push_back(r - 1);
                    val.push_back(-1.0);
                }
                if (j < n - 1) {
                    col.push_back(r + 1);
                    val.push_back(-1.0);
                }
                col.push_back(r);
                val.push_back(4.0);
                A.set(val, row, col);
                row[0] = r + 1;
            }
        return A;
    }

} // namespace

TEST_CASE("parallel spmv: mult matches the serial result (2D Poisson)", "[la]")
{
    auto A = make_2d_poisson(12); // 144 dofs.
    const std::int32_t N = 144;

    la::Vector<double> x(imap(N), 1);
    for (std::int32_t i = 0; i < N; ++i)
        x[i] = std::sin(0.1 * i);

    // Serial reference.
    la::Vector<double> y_serial(imap(N), 1);
    y_serial.set(0);
    {
        tbb::global_control gc(tbb::global_control::max_allowed_parallelism, 1);
        A.mult(x, y_serial);
    }

    // Parallel (default thread count).
    la::Vector<double> y_par(imap(N), 1);
    y_par.set(0);
    A.mult(x, y_par);

    for (std::int32_t i = 0; i < N; ++i)
        REQUIRE(y_par[i] == Catch::Approx(y_serial[i]).margin(1e-12));
}

TEST_CASE("parallel spmvT: multT matches the serial result", "[la]")
{
    auto A = make_2d_poisson(10);
    const std::int32_t N = 100;

    la::Vector<double> x(imap(N), 1);
    for (std::int32_t i = 0; i < N; ++i)
        x[i] = std::cos(0.05 * i);

    la::Vector<double> y_serial(imap(N), 1);
    y_serial.set(0);
    {
        tbb::global_control gc(tbb::global_control::max_allowed_parallelism, 1);
        A.multT(x, y_serial);
    }

    la::Vector<double> y_par(imap(N), 1);
    y_par.set(0);
    A.multT(x, y_par);

    for (std::int32_t i = 0; i < N; ++i)
        REQUIRE(y_par[i] == Catch::Approx(y_serial[i]).margin(1e-12));
}

TEST_CASE("parallel spmv: blocked matrix matches the serial result", "[la]")
{
    // 2x2-blocked 2x2 with coupling (bs=2), as in the preconditioner tests.
    auto m0 = imap(2), m1 = imap(2);
    la::SparsityPattern pattern(
        std::array<std::shared_ptr<const common::IndexMap>, 2> {m0, m1},
        std::array<int, 2> {2, 2});
    pattern.insert(0, 0);
    pattern.insert(1, 1);
    pattern.insert(0, 1);
    pattern.finalize();
    la::MatrixCSR<double> A(pattern);
    std::vector<std::int32_t> r0 {0}, r1 {1}, c0 {0}, c1 {1};
    A.set<2, 2>(std::vector<double> {4, 1, 2, 5}, r0, c0);
    A.set<2, 2>(std::vector<double> {6, 3, 7, 8}, r1, c1);
    A.set<2, 2>(std::vector<double> {1, 0, 0, 1}, r0, c1);

    la::Vector<double> x(imap(2), 2);
    for (std::size_t i = 0; i < 4; ++i)
        x[i] = static_cast<double>(i + 1);

    la::Vector<double> y_serial(imap(2), 2);
    y_serial.set(0);
    {
        tbb::global_control gc(tbb::global_control::max_allowed_parallelism, 1);
        A.mult(x, y_serial);
    }
    la::Vector<double> y_par(imap(2), 2);
    y_par.set(0);
    A.mult(x, y_par);

    for (std::size_t i = 0; i < 4; ++i)
        REQUIRE(y_par[i] == Catch::Approx(y_serial[i]).margin(1e-12));
}
