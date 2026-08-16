// hellofem::app — muparser expression wrapper tests
// SPDX-License-Identifier: MIT

#include "Expression.h"
#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"

#include <algorithm>
#include <unordered_map>

using hellofem::app::Expression;
using Catch::Approx;

TEST_CASE("Expression evaluates params and coordinates", "[app][expr]")
{
    double Vtot = 0.02;
    std::unordered_map<std::string, double*> vars;
    vars["Vtot"] = &Vtot;
    Expression e;
    e.parse("Vtot * x / 0.1", vars);
    REQUIRE(e.eval(0.05, 0, 0, 0) == Approx(0.01));
    REQUIRE(e.eval(0.1, 0, 0, 0) == Approx(0.02));

    // x/y/z/t are built-in.
    Expression e2;
    e2.parse("x*y + z*t", vars);
    REQUIRE(e2.eval(2, 3, 4, 5) == Approx(26.0));
}

TEST_CASE("Expression normalizes unit literals", "[app][expr]")
{
    std::unordered_map<std::string, double*> vars;
    Expression e;
    e.parse("20[mV]", vars);
    REQUIRE(e.eval(0, 0, 0, 0) == Approx(0.02));

    // Non-numeric expression referencing params is left to muparser.
    double htc = 5.0;
    double T = 303.15;
    vars["htc"] = &htc;
    vars["T"] = &T;
    Expression e2;
    e2.parse("htc * (T - 293.15)", vars);
    REQUIRE(e2.eval(0, 0, 0, 0) == Approx(50.0));
}

TEST_CASE("Expression reports used variables", "[app][expr]")
{
    double k = 2.0;
    std::unordered_map<std::string, double*> vars;
    vars["k"] = &k;
    Expression e;
    e.parse("k * sin(x) + y", vars);
    auto vs = e.variables();
    REQUIRE(std::find(vs.begin(), vs.end(), "k") != vs.end());
    REQUIRE(std::find(vs.begin(), vs.end(), "x") != vs.end());
}
