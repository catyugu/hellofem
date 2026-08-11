#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "basis/element-families.h"
#include "basis/finite-element.h"

#include <cmath>
#include <vector>

namespace b = hellofem::basis;

namespace {
    b::FiniteElement<double> lagrange(b::cell::type cell, int degree)
    {
        return b::create_element<double>(b::element::family::P, cell, degree,
            b::element::lagrange_variant::equispaced, b::element::dpc_variant::unset,
            false);
    }

    // Tabulate basis values (nd = 0) at `points` (row-major, `gdim` columns).
    // The returned mdspan points into `out`, which must outlive it.
    b::impl::mdspan_t<double, 4> tabulate(const b::FiniteElement<double>& e,
        const std::vector<double>& points, std::size_t gdim, std::vector<double>& out)
    {
        const std::size_t np = points.size() / gdim;
        const auto shape = e.tabulate_shape(0, np);
        out.assign(shape[0] * shape[1] * shape[2] * shape[3], 0.0);
        b::impl::mdspan_t<const double, 2> x(points.data(), np, gdim);
        b::impl::mdspan_t<double, 4> bv(out.data(), shape[0], shape[1], shape[2], shape[3]);
        e.tabulate(0, x, bv);
        return bv;
    }
} // namespace

TEST_CASE("Lagrange dimensions", "[basis][lagrange]")
{
    REQUIRE(lagrange(b::cell::type::interval, 1).dim() == 2);
    REQUIRE(lagrange(b::cell::type::interval, 2).dim() == 3);
    REQUIRE(lagrange(b::cell::type::interval, 3).dim() == 4);
    REQUIRE(lagrange(b::cell::type::triangle, 1).dim() == 3);
    REQUIRE(lagrange(b::cell::type::triangle, 2).dim() == 6);
    REQUIRE(lagrange(b::cell::type::triangle, 3).dim() == 10);
    REQUIRE(lagrange(b::cell::type::tetrahedron, 1).dim() == 4);
    REQUIRE(lagrange(b::cell::type::tetrahedron, 2).dim() == 10);
    REQUIRE(lagrange(b::cell::type::quadrilateral, 1).dim() == 4);
    REQUIRE(lagrange(b::cell::type::quadrilateral, 2).dim() == 9);
    REQUIRE(lagrange(b::cell::type::hexahedron, 1).dim() == 8);
    REQUIRE(lagrange(b::cell::type::hexahedron, 2).dim() == 27);
    REQUIRE(lagrange(b::cell::type::prism, 1).dim() == 6);
    REQUIRE(lagrange(b::cell::type::pyramid, 1).dim() == 5);
    REQUIRE(lagrange(b::cell::type::pyramid, 2).dim() == 14);
}

TEST_CASE("Lagrange P1 interval: known values", "[basis][lagrange]")
{
    auto e = lagrange(b::cell::type::interval, 1);
    std::vector<double> data;
    auto B = tabulate(e, {0.25}, 1, data);
    // phi_0 = 1 - x, phi_1 = x
    REQUIRE(B(0, 0, 0, 0) == Catch::Approx(0.75).margin(1e-12));
    REQUIRE(B(0, 0, 1, 0) == Catch::Approx(0.25).margin(1e-12));
}

TEST_CASE("Lagrange partition of unity", "[basis][lagrange]")
{
    auto e = lagrange(b::cell::type::triangle, 2);
    std::vector<double> data;
    auto B = tabulate(e, {0.2, 0.3, 0.5, 0.1, 0.7, 0.2, 0.1, 0.1}, 2, data);
    for (std::size_t p = 0; p < 4; ++p) {
        double sum = 0;
        for (std::size_t j = 0; j < 6; ++j)
            sum += B(0, p, j, 0);
        REQUIRE(sum == Catch::Approx(1.0).margin(1e-12));
    }
}

TEST_CASE("Lagrange dof-point identity", "[basis][lagrange]")
{
    // Tabulating the basis at its dof points gives a permutation matrix:
    // each row has exactly one 1 (each basis function is 1 at one dof point
    // and 0 at the others), and each column is used exactly once. This holds
    // regardless of the order in which basix numbers the dofs.
    auto check = [](const b::FiniteElement<double>& e,
                     const std::vector<double>& pts, std::size_t gdim) {
        const std::size_t n = static_cast<std::size_t>(e.dim());
        std::vector<double> data;
        auto B = tabulate(e, pts, gdim, data);
        std::vector<std::size_t> col_used(n, 0);
        for (std::size_t p = 0; p < n; ++p) {
            int hits = 0;
            for (std::size_t j = 0; j < n; ++j) {
                const double v = B(0, p, j, 0);
                const bool ok
                    = std::abs(v - 1.0) < 1e-12 || std::abs(v) < 1e-12;
                REQUIRE(ok);
                if (std::abs(v - 1.0) < 1e-12) {
                    ++hits;
                    ++col_used[j];
                }
            }
            REQUIRE(hits == 1);
        }
        for (const auto c : col_used)
            REQUIRE(c == 1);
    };

    check(lagrange(b::cell::type::triangle, 1), {0, 0, 1, 0, 0, 1}, 2);
    check(lagrange(b::cell::type::triangle, 2),
        {0, 0, 1, 0, 0, 1, 0.5, 0, 0.5, 0.5, 0, 0.5}, 2);
    check(lagrange(b::cell::type::tetrahedron, 1),
        {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1}, 3);
}
