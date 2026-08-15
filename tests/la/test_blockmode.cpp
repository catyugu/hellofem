// hellofem::la — BlockMode::expanded and scalar-view tests
// SPDX-License-Identifier: MIT

#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "common/IndexMap.h"
#include "la/MatrixCSR.h"
#include "la/SparsityPattern.h"
#include "la/Vector.h"

#include <cstdint>
#include <memory>
#include <vector>

using namespace hellofem;

namespace {

    std::shared_ptr<common::IndexMap> imap(std::int32_t n)
    {
        return std::make_shared<common::IndexMap>(0, n);
    }

    /// Build a 2x2-blocked 2x2 matrix of blocks (n=2 rows, scalar 2x2
    /// index space, bs=2 => 4 physical dofs):
    ///   [ [4 1] [0 0] ]
    ///   [ [2 5] [0 0] ]
    ///   [ [0 0] [6 3] ]
    ///   [ [0 0] [7 8] ]
    la::MatrixCSR<double> make_blocked(std::int32_t n)
    {
        la::SparsityPattern pattern(imap(n), 2);
        std::vector<std::int32_t> diag(n);
        for (std::int32_t i = 0; i < n; ++i)
            diag[i] = i;
        pattern.insert_diagonal(diag);
        pattern.finalize();

        la::MatrixCSR<double> A(pattern);
        // Block (0,0) = [[4,1],[2,5]], block (1,1) = [[6,3],[7,8]].
        std::vector<std::int32_t> r0 {0}, r1 {1};
        std::vector<std::int32_t> c0 {0}, c1 {1};
        A.set<2, 2>(std::vector<double> {4, 1, 2, 5}, r0, c0);
        A.set<2, 2>(std::vector<double> {6, 3, 7, 8}, r1, c1);
        return A;
    }

} // namespace

TEST_CASE("MatrixCSR: expanded mode constructs and matches compact dense", "[la]")
{
    la::MatrixCSR<double> A = make_blocked(2);

    // Compact dense.
    const auto dense_compact = A.to_dense();

    // Expanded mode must construct without throwing and match compact.
    la::SparsityPattern pattern(imap(2), 2);
    std::vector<std::int32_t> diag {0, 1};
    pattern.insert_diagonal(diag);
    pattern.finalize();
    la::MatrixCSR<double> B(pattern, la::BlockMode::expanded);
    REQUIRE(B.block_size() == std::array<int, 2> {1, 1});
    REQUIRE(B.num_owned_rows() == 4);
    REQUIRE(B.num_all_rows() == 4);

    // Set the same values in the expanded (scalar) matrix.
    std::vector<std::int32_t> rows {0, 1, 2, 3};
    std::vector<std::int32_t> cols {0, 1, 2, 3};
    std::vector<double> vals {4, 1, 0, 0, 2, 5, 0, 0, 0, 0, 6, 3, 0, 0, 7, 8};
    for (std::int32_t r = 0; r < 4; ++r)
        for (std::int32_t c = 0; c < 4; ++c)
            if (vals[static_cast<std::size_t>(r * 4 + c)] != 0)
                B.set(std::vector<double> {vals[static_cast<std::size_t>(r * 4 + c)]},
                    std::vector<std::int32_t> {r}, std::vector<std::int32_t> {c});

    const auto dense_expanded = B.to_dense();
    REQUIRE(dense_expanded.size() == dense_compact.size());
    for (std::size_t i = 0; i < dense_expanded.size(); ++i)
        REQUIRE(dense_expanded[i] == Catch::Approx(dense_compact[i]));
}

TEST_CASE("MatrixCSR: to_scalar() gives a correct scalar CSR view", "[la]")
{
    la::MatrixCSR<double> A = make_blocked(2);

    // Apply A to a vector and compare with the scalar view applied to the
    // same flattened vector.
    la::Vector<double> x(imap(2), 2);
    for (std::size_t i = 0; i < 4; ++i)
        x[i] = static_cast<double>(i + 1); // (1,2,3,4)

    la::Vector<double> y_compact(imap(2), 2);
    y_compact.set(0);
    A.mult(x, y_compact);

    la::Vector<double> y_scalar(imap(4), 1);
    y_scalar.set(0);
    // A.to_scalar() should give the 4x4 scalar matrix.
    auto S = A.to_scalar();
    REQUIRE(S.block_size() == std::array<int, 2> {1, 1});
    la::Vector<double> xf(imap(4), 1);
    for (std::size_t i = 0; i < 4; ++i)
        xf[i] = x[i];
    S.mult(xf, y_scalar);

    for (std::size_t i = 0; i < 4; ++i)
        REQUIRE(y_scalar[i] == Catch::Approx(y_compact[i]).margin(1e-12));
}
