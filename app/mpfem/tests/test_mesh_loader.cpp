// hellofem::app — mesh loader tests (mphtxt + boundary id normalization)
// SPDX-License-Identifier: MIT

#include "catch2/catch_test_macros.hpp"
#include "mesh_loader.h"

#include <filesystem>

using hellofem::app::load_mphtxt_mesh;

TEST_CASE("load_mphtxt_mesh normalizes boundary ids (+1) on the busbar", "[app][mesh]")
{
    const auto ref = std::filesystem::path(HELLOFEM_SOURCE_DIR)
        / ".cache/ref/mpfem/cases/busbar_steady/mesh.mphtxt";
    if (!std::filesystem::exists(ref)) {
        WARN("reference mesh not available; skipping");
        return;
    }

    auto lm = load_mphtxt_mesh(ref);
    REQUIRE(lm.mesh != nullptr);
    REQUIRE(lm.num_domains == 7);
    REQUIRE(lm.num_boundaries == 43);

    // Every boundary facet has a 1-based id in [1,43].
    REQUIRE(lm.facet_tags != nullptr);
    int mn = 1000, mx = 0;
    for (int v : lm.facet_tags->values()) {
        mn = std::min(mn, v);
        mx = std::max(mx, v);
    }
    REQUIRE(mn == 1);
    REQUIRE(mx == 43);

    // Domain ids in [1,7].
    REQUIRE(lm.cell_tags != nullptr);
    mn = 1000;
    mx = 0;
    for (int v : lm.cell_tags->values()) {
        mn = std::min(mn, v);
        mx = std::max(mx, v);
    }
    REQUIRE(mn == 1);
    REQUIRE(mx == 7);
}
