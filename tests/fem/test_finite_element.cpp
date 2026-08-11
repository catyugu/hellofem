// hellofem::fem — FiniteElement wrapper unit tests
// SPDX-License-Identifier: MIT

#include "basis/finite-element.h"
#include "fem/FiniteElement.h"
#include "mesh/cell_types.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <vector>

using namespace hellofem;

namespace {
    using B = basis::element::family;
    using C = basis::cell::type;
    using LV = basis::element::lagrange_variant;
    using DV = basis::element::dpc_variant;

    /// Build a fem::FiniteElement by value from a basis family/cell/degree.
    fem::FiniteElement<double> make_element(B family, C cell, int degree)
    {
        return fem::FiniteElement<double>(basis::create_element<double>(
            family, cell, degree, LV::equispaced, DV::unset, false));
    }
} // namespace

TEST_CASE("FiniteElement: Lagrange P1 scalar", "[fem]")
{
    fem::FiniteElement<double> e = make_element(B::P, C::triangle, 1);

    REQUIRE(e.space_dimension() == 3);
    REQUIRE(e.block_size() == 1);
    REQUIRE(e.value_size() == 1);
    REQUIRE(e.value_shape().empty());
    REQUIRE(e.reference_value_size() == 1);
    REQUIRE(e.cell_type() == mesh::CellType::triangle);
    REQUIRE(e.is_mixed() == false);
    REQUIRE(e.num_sub_elements() == 0);
    REQUIRE(e.interpolation_ident());
    REQUIRE(e.map_ident());
    REQUIRE(e.needs_dof_permutations() == false);
    REQUIRE(e.needs_dof_transformations() == false);
    REQUIRE(e.basix_element().dim() == 3);

    // One dof per vertex: entity_dofs[0][i] = {i}
    const auto& ed = e.entity_dofs();
    REQUIRE(ed.size() == 3); // dims 0,1,2
    REQUIRE(ed[0].size() == 3); // 3 vertices
    for (int i = 0; i < 3; ++i)
        REQUIRE(ed[0][i] == std::vector<int> {i});

    // Equality of identical elements
    fem::FiniteElement<double> e2 = make_element(B::P, C::triangle, 1);
    REQUIRE(e == e2);
    REQUIRE_FALSE(e != e2);

    // Signature identifies Lagrange
    REQUIRE(e.signature().find("Lagrange") != std::string::npos);

    // create_dof_layout for dofmap construction
    fem::ElementDofLayout layout = e.create_dof_layout();
    REQUIRE(layout.block_size() == 1);
    REQUIRE(layout.num_dofs() == 3);
}

TEST_CASE("FiniteElement: Lagrange P2 has edge dofs and permutes", "[fem]")
{
    fem::FiniteElement<double> e = make_element(B::P, C::triangle, 2);

    REQUIRE(e.space_dimension() == 6);
    REQUIRE(e.block_size() == 1);

    // Edge dofs: entity_dofs[1][i] = {3+i}
    const auto& ed = e.entity_dofs();
    REQUIRE(ed[1].size() == 3);
    for (int i = 0; i < 3; ++i)
        REQUIRE(ed[1][i] == std::vector<int> {3 + i});

    // P2 does not require dof transformations/permutations
    REQUIRE_FALSE(e.needs_dof_permutations());

    // permute/permute_inv round-trip is a no-op for P2
    std::vector<std::int32_t> dofs = {0, 1, 2, 3, 4, 5};
    std::vector<std::int32_t> copy = dofs;
    e.permute(copy, 0);
    REQUIRE(copy == dofs);
    e.permute_inv(copy, 0);
    REQUIRE(copy == dofs);
}

TEST_CASE("FiniteElement: Nedelec N1E uses covariant Piola map", "[fem]")
{
    fem::FiniteElement<double> e = make_element(B::N1E, C::triangle, 1);

    REQUIRE(e.space_dimension() == 3);
    REQUIRE(std::vector(e.value_shape().begin(), e.value_shape().end())
        == std::vector<std::size_t> {2});
    REQUIRE(e.value_size() == 2);
    REQUIRE(e.map_type() == basis::maps::type::covariantPiola);
    REQUIRE_FALSE(e.map_ident());
    // H(div)/H(curl) elements require dof transformations
    REQUIRE(e.needs_dof_transformations());
}

TEST_CASE("FiniteElement: Raviart-Thomas RT uses contravariant Piola map", "[fem]")
{
    fem::FiniteElement<double> e = make_element(B::RT, C::triangle, 1);

    REQUIRE(e.space_dimension() == 3);
    REQUIRE(e.map_type() == basis::maps::type::contravariantPiola);
    REQUIRE(e.needs_dof_transformations());

    // Interpolation operator maps dofs to point values: (3 dofs, 3 points x 2
    // components).
    const auto [op, shape] = e.interpolation_operator();
    REQUIRE(shape == std::array<std::size_t, 2> {3, 6});
    REQUIRE(op.size() == 18);

    const auto [pts, pshape] = e.interpolation_points();
    REQUIRE(pshape[0] == 3);
}

TEST_CASE("FiniteElement: create_interpolation_operator between P1 and P2", "[fem]")
{
    fem::FiniteElement<double> p1 = make_element(B::P, C::triangle, 1);
    fem::FiniteElement<double> p2 = make_element(B::P, C::triangle, 2);

    // P2 -> P1: shape (P1 dims, P2 dims) = (3, 6)
    const auto [op21, shape21] = p1.create_interpolation_operator(p2);
    REQUIRE(shape21 == std::array<std::size_t, 2> {3, 6});

    // P1 -> P2: shape (P2 dims, P1 dims) = (6, 3)
    const auto [op12, shape12] = p2.create_interpolation_operator(p1);
    REQUIRE(shape12 == std::array<std::size_t, 2> {6, 3});

    // Identical elements give the identity matrix
    fem::FiniteElement<double> p1b = make_element(B::P, C::triangle, 1);
    const auto [opid, shapeid] = p1.create_interpolation_operator(p1b);
    REQUIRE(shapeid == std::array<std::size_t, 2> {3, 3});
    REQUIRE(opid[0] == Catch::Approx(1.0));
    REQUIRE(opid[4] == Catch::Approx(1.0));
    REQUIRE(opid[8] == Catch::Approx(1.0));
    REQUIRE(opid[1] == Catch::Approx(0.0));
}

TEST_CASE("FiniteElement: blocked element has block copies as sub-elements", "[fem]")
{
    // Build a scalar P1 and block it with value_shape {3} (a vector field).
    basis::FiniteElement<double> base = basis::create_element<double>(
        B::P, C::triangle, 1, LV::equispaced, DV::unset, false);
    fem::FiniteElement<double> e(base, std::vector<std::size_t> {3});

    REQUIRE(e.block_size() == 3);
    REQUIRE(e.space_dimension() == 9); // 3 base dofs * 3 components
    REQUIRE(std::vector(e.value_shape().begin(), e.value_shape().end())
        == std::vector<std::size_t> {3});
    REQUIRE(e.value_size() == 3);
    REQUIRE(e.num_sub_elements() == 3);
    REQUIRE_FALSE(e.is_mixed()); // blocked, not mixed

    // Each sub-element is the scalar P1
    const auto& subs = e.sub_elements();
    REQUIRE(subs.size() == 3);
    REQUIRE(subs[0]->space_dimension() == 3);
    REQUIRE(subs[0]->block_size() == 1);

    // create_dof_layout carries the block size
    fem::ElementDofLayout layout = e.create_dof_layout();
    REQUIRE(layout.block_size() == 3);
    REQUIRE(layout.num_dofs() == 3); // scalar dofs per cell
}

TEST_CASE("FiniteElement: mixed element composed of P1 and P2", "[fem]")
{
    auto p1 = std::make_shared<const fem::FiniteElement<double>>(
        make_element(B::P, C::triangle, 1));
    auto p2 = std::make_shared<const fem::FiniteElement<double>>(
        make_element(B::P, C::triangle, 2));
    fem::FiniteElement<double> mixed({p1, p2});

    REQUIRE(mixed.space_dimension() == 9); // 3 + 6
    REQUIRE(mixed.is_mixed());
    REQUIRE(mixed.num_sub_elements() == 2);

    auto sub0 = mixed.extract_sub_element({0});
    auto sub1 = mixed.extract_sub_element({1});
    REQUIRE(sub0->space_dimension() == 3);
    REQUIRE(sub1->space_dimension() == 6);

    // Value-shape and basix accessors throw for mixed elements
    REQUIRE_THROWS(mixed.value_shape());
    REQUIRE_THROWS(mixed.basix_element());
    REQUIRE_THROWS(mixed.interpolation_operator());
}

TEST_CASE("FiniteElement: quadrature element", "[fem]")
{
    std::vector<double> points = {0.25, 0.5, 0.25, 0.25, 0.25, 0.5};
    fem::FiniteElement<double> e(mesh::CellType::triangle, points, {2, 2});

    REQUIRE(e.space_dimension() == 2);
    REQUIRE(e.block_size() == 1);
    REQUIRE(e.cell_type() == mesh::CellType::triangle);
    REQUIRE(e.is_mixed() == false);
    REQUIRE(e.interpolation_ident());
    REQUIRE(e.map_ident());

    const auto [pts, pshape] = e.interpolation_points();
    REQUIRE(pshape == std::array<std::size_t, 2> {2, 2});
    REQUIRE(pts == points);
}
