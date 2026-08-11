// hellofem::fem + hellofem::mesh::cell_types — core tests
// SPDX-License-Identifier: MIT

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "fem/CoordinateElement.h"
#include "fem/ElementDofLayout.h"
#include "mesh/cell_types.h"

#include <array>
#include <vector>

using namespace hellofem;
using namespace hellofem::fem;
using namespace hellofem::mesh;
using Catch::Approx;

TEST_CASE("cell_types round-trips with basis")
{
    REQUIRE(to_string(CellType::triangle) == "triangle");
    REQUIRE(to_type("tetrahedron") == CellType::tetrahedron);
    REQUIRE(cell_dim(CellType::hexahedron) == 3);
    REQUIRE(cell_dim(CellType::triangle) == 2);
    REQUIRE(is_simplex(CellType::tetrahedron));
    REQUIRE_FALSE(is_simplex(CellType::hexahedron));
    REQUIRE(num_cell_vertices(CellType::tetrahedron) == 4);

    for (CellType c : {CellType::point, CellType::interval, CellType::triangle,
             CellType::tetrahedron, CellType::quadrilateral,
             CellType::hexahedron, CellType::prism,
             CellType::pyramid}) {
        REQUIRE(cell_type_from_basix_type(cell_type_to_basix_type(c)) == c);
    }

    REQUIRE(cell_num_entities(CellType::tetrahedron, 3) == 1);
    REQUIRE(cell_num_entities(CellType::tetrahedron, 2) == 4);
    REQUIRE(cell_num_entities(CellType::tetrahedron, 1) == 6);
    REQUIRE(cell_num_entities(CellType::tetrahedron, 0) == 4);
}

TEST_CASE("ElementDofLayout counts dofs per entity")
{
    // A P2 Lagrange triangle: 3 vertices (1 dof each), 3 edges (1 dof
    // each), 0 interior -> 6 dofs.
    std::vector<std::vector<std::vector<int>>> entity_dofs = {
        {{0}, {1}, {2}}, // vertices
        {{3}, {4}, {5}}, // edges
        {{}} // interior
    };
    std::vector<std::vector<std::vector<int>>> closure_dofs = entity_dofs;
    ElementDofLayout layout(1, entity_dofs, closure_dofs, {}, {});
    REQUIRE(layout.num_dofs() == 6);
    REQUIRE(layout.block_size() == 1);
    REQUIRE_FALSE(layout.is_view());
    REQUIRE(layout.entity_dofs(0, 1) == std::vector<int>({1}));
    REQUIRE(layout.entity_dofs(1, 0) == std::vector<int>({3}));
    REQUIRE(layout.entity_closure_dofs_all().size() == 3);

    // copy() clears the parent map
    ElementDofLayout copy = layout.copy();
    REQUIRE_FALSE(copy.is_view());
}

TEST_CASE("CoordinateElement evaluates the P1 geometry map")
{
    CoordinateElement<double> element(CellType::triangle, 1);
    REQUIRE(element.is_affine());
    REQUIRE(element.degree() == 1);
    REQUIRE(element.dim() == 3);
    REQUIRE(element.cell_shape() == CellType::triangle);

    // Tabulate basis at the reference vertices
    const int num_points = 3;
    auto shape = element.tabulate_shape(0, num_points);
    REQUIRE(shape[1] == num_points);
    std::vector<double> basis(shape[0] * shape[1] * shape[2] * shape[3]);
    std::vector<double> X {0.0, 0.0, 1.0, 0.0, 0.0, 1.0};
    element.tabulate(0, X, {3, 2}, basis);

    // Partition-of-unity at every point
    for (int p = 0; p < num_points; ++p) {
        double sum = 0;
        for (int b = 0; b < element.dim(); ++b)
            sum += basis[p * element.dim() + b];
        REQUIRE(sum == Approx(1.0));
    }

    // create_dof_layout for P1 triangle: 3 vertex dofs
    auto layout = element.create_dof_layout();
    REQUIRE(layout.num_dofs() == 3);

    // push_forward maps reference to physical coordinates
    // cell nodes: (0,0), (1,0), (0,1); at reference (0.5, 0.5) the
    // barycentric weights are (0, 0.5, 0.5).
    std::vector<double> geometry {0.0, 0.0, 1.0, 0.0, 0.0, 1.0};
    std::vector<double> x(2, 0.0);
    md::mdspan<double, md::dextents<std::size_t, 2>> xm(x.data(), 1, 2);
    md::mdspan<const double, md::dextents<std::size_t, 2>> gm(geometry.data(),
        3, 2);
    std::vector<double> phi {0.0, 0.5, 0.5};
    md::mdspan<const double, md::dextents<std::size_t, 2>> pm(phi.data(), 1, 3);
    CoordinateElement<double>::push_forward(xm, gm, pm);
    REQUIRE(xm(0, 0) == Approx(0.5));
    REQUIRE(xm(0, 1) == Approx(0.5));
}
