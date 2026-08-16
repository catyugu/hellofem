// hellofem::app — unit parser tests
// SPDX-License-Identifier: MIT

#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "units.h"

using hellofem::app::parse_si;
using hellofem::app::parse_unit;
using Catch::Approx;

TEST_CASE("parse_unit converts to SI", "[app][units]")
{
    REQUIRE(parse_unit("m") == Approx(1.0));
    REQUIRE(parse_unit("cm") == Approx(1e-2));
    REQUIRE(parse_unit("mm") == Approx(1e-3));
    REQUIRE(parse_unit("GPa") == Approx(1e9));
    REQUIRE(parse_unit("MPa") == Approx(1e6));
    REQUIRE(parse_unit("mV") == Approx(1e-3));
    REQUIRE(parse_unit("W/(m*K)") == Approx(1.0));
    REQUIRE(parse_unit("kg/m^3") == Approx(1.0));
    REQUIRE(parse_unit("1/K") == Approx(1.0));
}

TEST_CASE("parse_si handles bare numbers and unit literals", "[app][units]")
{
    REQUIRE(parse_si("0.006") == Approx(0.006));
    REQUIRE(parse_si("20[mV]") == Approx(0.02));
    REQUIRE(parse_si("5[W/m^2/K]") == Approx(5.0));
    REQUIRE(parse_si("110[GPa]") == Approx(1.1e11));
    REQUIRE(parse_si("293.15[K]") == Approx(293.15));
    REQUIRE(parse_si("7.407e5[S/m]") == Approx(740700.0));
}
