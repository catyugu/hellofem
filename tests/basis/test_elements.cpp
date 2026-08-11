#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "basis/element-families.h"
#include "basis/finite-element.h"

#include <vector>

namespace b = hellofem::basis;

namespace {
    b::FiniteElement<double> make(b::element::family f, b::cell::type cell, int deg)
    {
        // CR/bubble/Hermite/Regge/HHJ do not accept a Lagrange variant.
        const bool uses_lvariant
            = (f != b::element::family::CR && f != b::element::family::bubble
                && f != b::element::family::Hermite && f != b::element::family::Regge
                && f != b::element::family::HHJ);
        const auto lvar = uses_lvariant ? b::element::lagrange_variant::equispaced
                                        : b::element::lagrange_variant::unset;
        return b::create_element<double>(f, cell, deg, lvar,
            b::element::dpc_variant::unset, false);
    }
} // namespace

TEST_CASE("Vector element dof counts", "[basis][elements]")
{
    REQUIRE(make(b::element::family::RT, b::cell::type::triangle, 1).dim() == 3);
    REQUIRE(make(b::element::family::RT, b::cell::type::triangle, 2).dim() == 8);
    REQUIRE(make(b::element::family::RT, b::cell::type::tetrahedron, 1).dim() == 4);
    REQUIRE(make(b::element::family::RT, b::cell::type::tetrahedron, 2).dim() == 15);
    REQUIRE(make(b::element::family::N1E, b::cell::type::triangle, 1).dim() == 3);
    REQUIRE(make(b::element::family::N1E, b::cell::type::triangle, 2).dim() == 8);
    REQUIRE(make(b::element::family::N1E, b::cell::type::tetrahedron, 1).dim() == 6);
    REQUIRE(make(b::element::family::BDM, b::cell::type::triangle, 1).dim() == 6);
    REQUIRE(make(b::element::family::BDM, b::cell::type::triangle, 2).dim() == 12);
    REQUIRE(make(b::element::family::BDM, b::cell::type::tetrahedron, 1).dim() == 12);
    REQUIRE(make(b::element::family::N2E, b::cell::type::triangle, 1).dim() == 6);
    REQUIRE(make(b::element::family::CR, b::cell::type::triangle, 1).dim() == 3);
    REQUIRE(make(b::element::family::bubble, b::cell::type::triangle, 3).dim() == 1);
    REQUIRE(make(b::element::family::bubble, b::cell::type::triangle, 4).dim() == 3);
    REQUIRE(make(b::element::family::serendipity, b::cell::type::quadrilateral, 2).dim() == 8);
    REQUIRE(make(b::element::family::Hermite, b::cell::type::interval, 3).dim() == 4);
    REQUIRE(make(b::element::family::Regge, b::cell::type::triangle, 1).dim() == 9);
    REQUIRE(make(b::element::family::HHJ, b::cell::type::triangle, 1).dim() == 9);
}

TEST_CASE("Element value shapes", "[basis][elements]")
{
    REQUIRE(make(b::element::family::P, b::cell::type::triangle, 1).value_shape().empty());
    REQUIRE(make(b::element::family::CR, b::cell::type::triangle, 1).value_shape().empty());
    REQUIRE(make(b::element::family::RT, b::cell::type::triangle, 1).value_shape()
        == std::vector<std::size_t> {2});
    REQUIRE(make(b::element::family::N1E, b::cell::type::tetrahedron, 1).value_shape()
        == std::vector<std::size_t> {3});
    REQUIRE(make(b::element::family::Regge, b::cell::type::triangle, 1).value_shape()
        == std::vector<std::size_t> {2, 2});
    REQUIRE(make(b::element::family::HHJ, b::cell::type::triangle, 1).value_shape()
        == std::vector<std::size_t> {2, 2});
}
