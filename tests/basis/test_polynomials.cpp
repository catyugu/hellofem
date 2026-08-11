#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "basis/cell.h"
#include "basis/lattice.h"
#include "basis/polynomials.h"

namespace b = hellofem::basis;

TEST_CASE("Polynomial space dimensions", "[basis][polynomials]")
{
    REQUIRE(b::polynomials::dim(b::polynomials::type::legendre, b::cell::type::interval, 2) == 3);
    REQUIRE(b::polynomials::dim(b::polynomials::type::legendre, b::cell::type::triangle, 2) == 6);
    REQUIRE(b::polynomials::dim(b::polynomials::type::legendre, b::cell::type::tetrahedron, 1) == 4);
    REQUIRE(b::polynomials::dim(b::polynomials::type::bernstein, b::cell::type::interval, 2) == 3);
    REQUIRE(b::polynomials::dim(b::polynomials::type::lagrange, b::cell::type::interval, 2) == 3);
}

TEST_CASE("Lattice equidistant points", "[basis][lattice]")
{
    // interval, n = 3, exterior -> 4 points {0, 1/3, 2/3, 1}
    {
        auto [pts, shape] = b::lattice::create<double>(
            b::cell::type::interval, 3, b::lattice::type::equispaced, true);
        REQUIRE(shape[0] == 4);
        REQUIRE(pts.size() == 4);
        REQUIRE(pts[0] == Catch::Approx(0.0).margin(1e-12));
        REQUIRE(pts[3] == Catch::Approx(1.0).margin(1e-12));
    }
    // triangle, n = 2, exterior -> 6 points in [0,1]^2
    {
        auto [pts, shape] = b::lattice::create<double>(
            b::cell::type::triangle, 2, b::lattice::type::equispaced, true);
        REQUIRE(shape[0] == 6);
        REQUIRE(pts.size() == 12);
        for (double v : pts) {
            REQUIRE(v >= -1e-12);
            REQUIRE(v <= 1.0 + 1e-12);
        }
    }
}
