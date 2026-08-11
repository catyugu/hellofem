// hellofem::la — SparsityPattern unit tests
// SPDX-License-Identifier: MIT

#include "la/SparsityPattern.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <vector>

using namespace hellofem;

TEST_CASE("SparsityPattern: CSR arrays after finalize", "[la]")
{
    auto map = std::make_shared<common::IndexMap>(0, 6);
    la::SparsityPattern pattern(map);

    // Diagonal + a handful of off-diagonal entries.
    std::vector<std::int32_t> diag = {0, 1, 2, 3, 4, 5};
    pattern.insert_diagonal(diag);
    pattern.insert(0, 1);
    pattern.insert(0, 2);
    pattern.insert(1, 2);
    pattern.insert(1, 3);
    pattern.insert(2, 4);
    pattern.insert(3, 5);
    pattern.insert(4, 5);

    pattern.finalize();

    const auto [cols, offsets] = pattern.graph();

    // Expected sorted, deduped rows:
    //   row 0: {0,1,2}  row 1: {1,2,3}  row 2: {2,4}
    //   row 3: {3,5}    row 4: {4,5}    row 5: {5}
    std::vector<std::int32_t> expected_cols
        = {0, 1, 2, 1, 2, 3, 2, 4, 3, 5, 4, 5, 5};
    std::vector<std::int64_t> expected_offsets
        = {0, 3, 6, 8, 10, 12, 13};

    REQUIRE(std::vector(cols.begin(), cols.end()) == expected_cols);
    REQUIRE(std::vector(offsets.begin(), offsets.end()) == expected_offsets);
    REQUIRE(pattern.num_nonzeros() == 13);
    REQUIRE(pattern.block_size(0) == 1);
    REQUIRE(pattern.block_size(1) == 1);
}

TEST_CASE("SparsityPattern: duplicate entries are deduplicated", "[la]")
{
    auto map = std::make_shared<common::IndexMap>(0, 3);
    la::SparsityPattern pattern(map);

    pattern.insert(0, 1);
    pattern.insert(0, 1); // duplicate
    pattern.insert(1, 0);
    std::vector<std::int32_t> diag = {0, 1, 2};
    pattern.insert_diagonal(diag);

    pattern.finalize();

    const auto [cols, offsets] = pattern.graph();
    // row 0: {0,1}  row 1: {0,1}  row 2: {2}
    std::vector<std::int32_t> expected_cols = {0, 1, 0, 1, 2};
    std::vector<std::int64_t> expected_offsets = {0, 2, 4, 5};
    REQUIRE(std::vector(cols.begin(), cols.end()) == expected_cols);
    REQUIRE(std::vector(offsets.begin(), offsets.end()) == expected_offsets);
    REQUIRE(pattern.num_nonzeros() == 5);
}

TEST_CASE("SparsityPattern: block size is metadata only", "[la]")
{
    auto map = std::make_shared<common::IndexMap>(0, 3);
    la::SparsityPattern pattern(map, 2);

    std::vector<std::int32_t> diag = {0, 1, 2};
    pattern.insert_diagonal(diag);

    REQUIRE(pattern.block_size(0) == 2);
    REQUIRE(pattern.block_size(1) == 2);

    pattern.finalize();

    // Block size is not expanded into the pattern; entries stay at
    // scalar-block indices.
    const auto [cols, offsets] = pattern.graph();
    std::vector<std::int32_t> expected_cols = {0, 1, 2};
    std::vector<std::int64_t> expected_offsets = {0, 1, 2, 3};
    REQUIRE(std::vector(cols.begin(), cols.end()) == expected_cols);
    REQUIRE(std::vector(offsets.begin(), offsets.end()) == expected_offsets);
    REQUIRE(pattern.num_nonzeros() == 3);
}

TEST_CASE("SparsityPattern: insert after finalize throws", "[la]")
{
    auto map = std::make_shared<common::IndexMap>(0, 2);
    la::SparsityPattern pattern(map);

    std::vector<std::int32_t> diag = {0, 1};
    pattern.insert_diagonal(diag);
    pattern.finalize();

    REQUIRE_THROWS(pattern.insert(0, 0));
    std::vector<std::int32_t> d2 = {0};
    REQUIRE_THROWS(pattern.insert_diagonal(d2));

    REQUIRE(pattern.num_nonzeros() == 2);
}

TEST_CASE("SparsityPattern: rectangular pattern uses both maps", "[la]")
{
    auto rows = std::make_shared<common::IndexMap>(0, 3);
    auto cols = std::make_shared<common::IndexMap>(0, 4);
    la::SparsityPattern pattern({rows, cols}, {1, 1});

    std::vector<std::int32_t> r = {0, 1};
    std::vector<std::int32_t> col_vals = {0, 2, 3};
    pattern.insert(r, col_vals);

    pattern.finalize();

    const auto [cc, o] = pattern.graph();
    // row 0: {0,2,3}  row 1: {0,2,3}  row 2: {}
    std::vector<std::int32_t> expected_cols = {0, 2, 3, 0, 2, 3};
    std::vector<std::int64_t> expected_offsets = {0, 3, 6, 6};
    REQUIRE(std::vector(cc.begin(), cc.end()) == expected_cols);
    REQUIRE(std::vector(o.begin(), o.end()) == expected_offsets);
    REQUIRE(pattern.index_map(0) == rows);
    REQUIRE(pattern.index_map(1) == cols);
}
