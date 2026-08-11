#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "basis/cell.h"
#include "basis/element-families.h"
#include "basis/finite-element.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

namespace b = hellofem::basis;

namespace {

    struct Combo {
        b::element::family family;
        b::cell::type cell;
        int degree;
    };

    // Curated list of supported (family, cell, degree) combinations.
    const std::vector<Combo> combos = {
        // Lagrange on every reference cell
        {b::element::family::P, b::cell::type::interval, 1},
        {b::element::family::P, b::cell::type::interval, 2},
        {b::element::family::P, b::cell::type::triangle, 1},
        {b::element::family::P, b::cell::type::triangle, 3},
        {b::element::family::P, b::cell::type::tetrahedron, 1},
        {b::element::family::P, b::cell::type::tetrahedron, 2},
        {b::element::family::P, b::cell::type::quadrilateral, 2},
        {b::element::family::P, b::cell::type::hexahedron, 1},
        {b::element::family::P, b::cell::type::hexahedron, 2},
        {b::element::family::P, b::cell::type::prism, 1},
        {b::element::family::P, b::cell::type::pyramid, 1},
        // H(div) / H(curl) on simplices
        {b::element::family::RT, b::cell::type::triangle, 1},
        {b::element::family::RT, b::cell::type::triangle, 2},
        {b::element::family::RT, b::cell::type::tetrahedron, 1},
        {b::element::family::RT, b::cell::type::tetrahedron, 2},
        // RT on tensor-product cells dispatches to RTC
        {b::element::family::RT, b::cell::type::quadrilateral, 1},
        {b::element::family::RT, b::cell::type::hexahedron, 1},
        {b::element::family::N1E, b::cell::type::triangle, 1},
        {b::element::family::N1E, b::cell::type::triangle, 2},
        {b::element::family::N1E, b::cell::type::tetrahedron, 1},
        {b::element::family::N2E, b::cell::type::triangle, 1},
        {b::element::family::N2E, b::cell::type::tetrahedron, 1},
        {b::element::family::BDM, b::cell::type::triangle, 1},
        {b::element::family::BDM, b::cell::type::tetrahedron, 1},
        // Discontinuous / bubble / Hermite / Regge / HHJ
        {b::element::family::CR, b::cell::type::triangle, 1},
        {b::element::family::CR, b::cell::type::tetrahedron, 1},
        {b::element::family::bubble, b::cell::type::interval, 2},
        {b::element::family::bubble, b::cell::type::triangle, 3},
        {b::element::family::bubble, b::cell::type::tetrahedron, 4},
        {b::element::family::bubble, b::cell::type::hexahedron, 2},
        {b::element::family::Hermite, b::cell::type::interval, 3},
        {b::element::family::Regge, b::cell::type::triangle, 1},
        {b::element::family::Regge, b::cell::type::tetrahedron, 1},
        {b::element::family::HHJ, b::cell::type::triangle, 1},
        {b::element::family::HHJ, b::cell::type::tetrahedron, 1},
        // Tensor-product cells
        {b::element::family::serendipity, b::cell::type::quadrilateral, 2},
        {b::element::family::serendipity, b::cell::type::hexahedron, 2},
        // N1E on tensor-product cells dispatches to NCE
        {b::element::family::N1E, b::cell::type::quadrilateral, 1},
        {b::element::family::N1E, b::cell::type::hexahedron, 1},
        {b::element::family::DPC, b::cell::type::hexahedron, 2},
    };

    b::FiniteElement<double> make(b::element::family f, b::cell::type cell, int deg)
    {
        // CR/bubble/Hermite/Regge/HHJ/DPC do not accept a Lagrange variant.
        const bool uses_lvariant
            = (f != b::element::family::CR && f != b::element::family::bubble
                && f != b::element::family::Hermite && f != b::element::family::Regge
                && f != b::element::family::HHJ && f != b::element::family::DPC);
        const auto lvar = uses_lvariant ? b::element::lagrange_variant::equispaced
                                        : b::element::lagrange_variant::unset;
        // DPC is a discontinuous space by construction and needs a variant.
        const bool discontinuous = (f == b::element::family::DPC);
        const auto dvar = (f == b::element::family::DPC)
            ? b::element::dpc_variant::simplex_equispaced
            : b::element::dpc_variant::unset;
        return b::create_element<double>(f, cell, deg, lvar, dvar, discontinuous);
    }

    // Sample points inside the reference cell: vertices plus the centroid.
    std::vector<double> sample_points(b::cell::type cell, std::size_t& gdim_out)
    {
        auto [verts, shape] = b::cell::geometry<double>(cell);
        const std::size_t nv = shape[0];
        const std::size_t gdim = shape[1];
        std::vector<double> pts;
        for (std::size_t i = 0; i < nv; ++i)
            for (std::size_t d = 0; d < gdim; ++d)
                pts.push_back(verts[i * gdim + d]);
        // Centroid
        for (std::size_t d = 0; d < gdim; ++d) {
            double c = 0.0;
            for (std::size_t i = 0; i < nv; ++i)
                c += verts[i * gdim + d];
            pts.push_back(c / static_cast<double>(nv));
        }
        gdim_out = gdim;
        return pts;
    }

} // namespace

TEST_CASE("All element families tabulate to finite values", "[basis][tabulate]")
{
    for (const auto& c : combos) {
        CAPTURE(static_cast<int>(c.family), static_cast<int>(c.cell), c.degree);
        b::FiniteElement<double> e = make(c.family, c.cell, c.degree);
        const std::size_t dim = e.dim();
        std::size_t gdim = 0;
        std::vector<double> pts = sample_points(c.cell, gdim);
        const std::size_t np = pts.size() / gdim;

        for (int nd : {0, 1}) {
            auto shape = e.tabulate_shape(nd, np);
            REQUIRE(shape[1] == np);
            REQUIRE(shape[2] == dim);
            REQUIRE(shape[3] > 0);
            std::vector<double> data(shape[0] * shape[1] * shape[2] * shape[3], 0.0);
            b::impl::mdspan_t<const double, 2> x(pts.data(), np, gdim);
            b::impl::mdspan_t<double, 4> B(data.data(), shape[0], shape[1], shape[2], shape[3]);
            e.tabulate(nd, x, B);
            for (double v : data)
                REQUIRE(std::isfinite(v));
        }
    }
}

TEST_CASE("Scalar nodal elements satisfy partition of unity", "[basis][tabulate]")
{
    // Only nodal (Lagrange-type) scalar spaces interpolate constants exactly:
    // P, CR (affine, values pinned at edge midpoints), serendipity.
    // bubble vanishes on the boundary, Hermite carries derivative dofs, and
    // DPC is a discontinuous space with a non-nodal basis, so none of these
    // sum to one.
    const std::array<b::element::family, 3> nu = {b::element::family::P,
        b::element::family::CR, b::element::family::serendipity};
    for (const auto& c : combos) {
        if (std::find(nu.begin(), nu.end(), c.family) == nu.end())
            continue;
        const b::FiniteElement<double> e = make(c.family, c.cell, c.degree);
        const std::vector<std::size_t> vs = e.value_shape();
        if (!vs.empty())
            continue; // only scalar-valued families
        CAPTURE(static_cast<int>(c.family), static_cast<int>(c.cell), c.degree);
        std::size_t gdim = 0;
        std::vector<double> pts = sample_points(c.cell, gdim);
        const std::size_t np = pts.size() / gdim;
        auto shape = e.tabulate_shape(0, np);
        std::vector<double> data(shape[0] * shape[1] * shape[2] * shape[3], 0.0);
        b::impl::mdspan_t<const double, 2> x(pts.data(), np, gdim);
        b::impl::mdspan_t<double, 4> B(data.data(), shape[0], shape[1], shape[2], shape[3]);
        e.tabulate(0, x, B);
        const std::size_t dim = e.dim();
        for (std::size_t p = 0; p < np; ++p) {
            double sum = 0.0;
            for (std::size_t j = 0; j < dim; ++j)
                sum += B(0, p, j, 0);
            REQUIRE(sum == Catch::Approx(1.0).margin(1e-10));
        }
    }
}

TEST_CASE("P1 interval gradient known values", "[basis][tabulate]")
{
    auto e = make(b::element::family::P, b::cell::type::interval, 1);
    std::vector<double> data;
    std::vector<double> pts = {0.25};
    auto shape = e.tabulate_shape(1, 1);
    data.assign(shape[0] * shape[1] * shape[2] * shape[3], 0.0);
    b::impl::mdspan_t<const double, 2> x(pts.data(), 1, 1);
    b::impl::mdspan_t<double, 4> B(data.data(), shape[0], shape[1], shape[2], shape[3]);
    e.tabulate(1, x, B);
    // d/dx phi_0 = -1, d/dx phi_1 = +1 on the reference interval
    REQUIRE(B(1, 0, 0, 0) == Catch::Approx(-1.0).margin(1e-12));
    REQUIRE(B(1, 0, 1, 0) == Catch::Approx(1.0).margin(1e-12));
}
