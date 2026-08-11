#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "basis/cell.h"
#include "basis/polyset.h"
#include "basis/quadrature.h"

namespace b = hellofem::basis;

TEST_CASE("Quadrature integrates reference cell volume", "[basis][quadrature]")
{
    // sum of weights = volume of the reference cell
    auto vol = [](b::cell::type c) {
        auto [pts, wts] = b::quadrature::make_quadrature<double>(
            b::quadrature::type::Default, c, b::polyset::type::standard, 4);
        double s = 0;
        for (double w : wts)
            s += w;
        return s;
    };
    REQUIRE(vol(b::cell::type::interval) == Catch::Approx(1.0).margin(1e-12));
    REQUIRE(vol(b::cell::type::triangle) == Catch::Approx(0.5).margin(1e-12));
    REQUIRE(vol(b::cell::type::tetrahedron) == Catch::Approx(1.0 / 6.0).margin(1e-12));
    REQUIRE(vol(b::cell::type::quadrilateral) == Catch::Approx(1.0).margin(1e-12));
    REQUIRE(vol(b::cell::type::hexahedron) == Catch::Approx(1.0).margin(1e-12));
    REQUIRE(vol(b::cell::type::prism) == Catch::Approx(0.5).margin(1e-12));
    REQUIRE(vol(b::cell::type::pyramid) == Catch::Approx(1.0 / 3.0).margin(1e-12));
}

TEST_CASE("Quadrature integrates a monomial on the triangle", "[basis][quadrature]")
{
    auto [pts, wts] = b::quadrature::make_quadrature<double>(
        b::quadrature::type::Default, b::cell::type::triangle,
        b::polyset::type::standard, 4);
    // integrate x^2 over the reference triangle: 1/12
    double s = 0;
    for (std::size_t i = 0; i < wts.size(); ++i)
        s += wts[i] * pts[2 * i] * pts[2 * i];
    REQUIRE(s == Catch::Approx(1.0 / 12.0).margin(1e-12));
}
