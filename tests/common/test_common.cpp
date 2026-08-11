// hellofem::common — unit tests
// SPDX-License-Identifier: MIT

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "common/IndexMap.h"
#include "common/Table.h"
#include "common/Timer.h"
#include "common/local_range.h"
#include "common/math.h"
#include "common/sort.h"
#include "common/timing.h"
#include "common/utils.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

using namespace hellofem;
using namespace hellofem::common;
using namespace hellofem::math;
using Catch::Approx;

TEST_CASE("local_range partitions contiguously")
{
    const std::int64_t N = 10;
    std::array<std::int64_t, 2> prev {0, 0};
    std::int64_t total = 0;
    for (int i = 0; i < 3; ++i) {
        auto r = local_range(i, N, 3);
        REQUIRE(r[0] == prev[1]);
        REQUIRE(r[1] >= r[0]);
        prev = r;
        total += r[1] - r[0];
    }
    REQUIRE(total == N);
}

TEST_CASE("IndexMap is a single-process identity block")
{
    IndexMap map(100, 5);
    REQUIRE(map.size_local() == 5);
    REQUIRE(map.size_global() == 105);
    REQUIRE(map.local_range() == std::array<std::int64_t, 2>({100, 105}));
    REQUIRE(map.num_ghosts() == 0);
    REQUIRE(map.ghosts().empty());
    REQUIRE(map.src().empty());
    REQUIRE(map.dest().empty());
    REQUIRE(map.owners().empty());

    // local_to_global / global_to_local round-trip
    std::vector<std::int32_t> local {0, 2, 4};
    std::vector<std::int64_t> global(local.size());
    map.local_to_global(local, global);
    REQUIRE(global == std::vector<std::int64_t>({100, 102, 104}));

    std::vector<std::int32_t> back(local.size());
    map.global_to_local(global, back);
    REQUIRE(back == local);

    std::vector<std::int64_t> foreign {104, 200};
    std::vector<std::int32_t> loc(foreign.size());
    map.global_to_local(foreign, loc);
    REQUIRE(loc == std::vector<std::int32_t>({4, -1}));

    REQUIRE(map.global_indices()
        == std::vector<std::int64_t>({100, 101, 102, 103, 104}));
}

TEST_CASE("create_sub_index_map renumbers a subset contiguously")
{
    IndexMap map(0, 10);
    std::vector<std::int32_t> indices {0, 2, 5, 7};
    auto [sub, sub_imap_to_imap] = create_sub_index_map(map, indices);
    REQUIRE(sub.size_local() == 4);
    REQUIRE(sub.local_range() == std::array<std::int64_t, 2>({0, 4}));
    REQUIRE(sub.size_global() == 4);
    REQUIRE(sub_imap_to_imap == indices);
}

TEST_CASE("radix_sort orders signed and unsigned keys")
{
    std::vector<int> v {3, -1, 0, 2, -5};
    radix_sort(v);
    REQUIRE(v == std::vector<int>({-5, -1, 0, 2, 3}));

    std::vector<std::uint32_t> u {9, 1, 7, 3};
    radix_sort(u);
    REQUIRE(u == std::vector<std::uint32_t>({1, 3, 7, 9}));
}

TEST_CASE("sort_by_perm sorts rows by leading columns")
{
    // Three rows of 2 columns: (3,9) (1,7) (2,8)
    std::vector<int> x {3, 9, 1, 7, 2, 8};
    auto perm = sort_by_perm<int>(x, 2);
    REQUIRE(perm == std::vector<std::int32_t>({1, 2, 0}));

    // Sort by first column only, ignoring the payload
    auto perm1 = sort_by_perm<int>(x, 2, 1);
    REQUIRE(perm1 == std::vector<std::int32_t>({1, 2, 0}));
}

TEST_CASE("sort_unique keeps smallest value per index")
{
    std::vector<int> idx {3, 1, 3, 2, 1};
    std::vector<double> val {1.5, 0.5, 0.5, 2.0, 1.0};
    auto [is, vs] = sort_unique(idx, val);
    REQUIRE(is == std::vector<int>({1, 2, 3}));
    REQUIRE(vs == std::vector<double>({0.5, 2.0, 0.5}));
}

TEST_CASE("math cross/det/inv/pinv")
{
    using array3 = std::array<double, 3>;
    REQUIRE(cross(array3 {1, 0, 0}, array3 {0, 1, 0})
        == array3 {0, 0, 1});

    std::array<double, 9> A {1, 2, 3, 0, 1, 4, 5, 6, 0};
    md::mdspan<double, md::dextents<std::size_t, 2>> Am(A.data(), 3, 3);
    REQUIRE(det(Am) == Approx(1.0));

    std::array<double, 9> B;
    md::mdspan<double, md::dextents<std::size_t, 2>> Bm(B.data(), 3, 3);
    inv(Am, Bm);
    // A * A^-1 == I
    std::array<double, 9> prod {0, 0, 0, 0, 0, 0, 0, 0, 0};
    md::mdspan<double, md::dextents<std::size_t, 2>> Pm(prod.data(), 3, 3);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            for (std::size_t k = 0; k < 3; ++k)
                Pm(i, j) += Am(i, k) * Bm(k, j);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            REQUIRE(Pm(i, j) == Approx(i == j ? 1.0 : 0.0).margin(1e-12));
}

TEST_CASE("Table stores and prints entries")
{
    Table table("Timings");
    table.set("Foo", "Assemble", 0.010);
    table.set("Foo", "Solve", 0.020);
    table.set("Bar", "Assemble", 0.011);
    table.set("Bar", "Solve", 0.019);
    REQUIRE(std::get<double>(table.get("Foo", "Solve")) == Approx(0.020));
    REQUIRE(table.name == "Timings");
    REQUIRE_FALSE(table.str().empty());
}

TEST_CASE("Timer measures a monotonic elapsed time")
{
    {
        Timer timer("common-test");
        timer.start();
        timer.stop();
        REQUIRE(timer.elapsed().count() >= 0.0);
        timer.flush(); // register in the logger explicitly
    }
    // flush() registered the timing in the singleton
    auto [count, wall] = timing("common-test");
    REQUIRE(count == 1);
    REQUIRE(wall.count() >= 0.0);
}
