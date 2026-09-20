// hellofem::app — physics solver tests: manufactured and analytic solutions
// SPDX-License-Identifier: MIT

#include "case_context.h"
#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "common/IndexMap.h"
#include "electric.h"
#include "fixture.h"
#include "heat.h"
#include "la/KrylovSolver.h"
#include "la/MatrixCSR.h"
#include "la/SparsityPattern.h"
#include "la/Vector.h"
#include "mesh/generation.h"
#include "mesh/utils.h"
#include "physics_field.h"
#include "solid.h"
#include "solver.h"

#include <array>
#include <cmath>
#include <numeric>
#include <random>
#include <string>
#include <tuple>
#include <vector>

namespace la = hellofem::la;

using Catch::Approx;
using namespace hellofem::app;
using hellofem::app::test::constant_facet_property;
using hellofem::app::test::constant_property;
using hellofem::app::test::make_box_fixture;

TEST_CASE("DomainProperty: sparse domains and field-dependent values", "[app][physics]")
{
    // 3x1x1 box: cell 1 belongs to domain 2, cell 2 to domain 1, cell 0 to
    // neither (its value stays at the initial zero).
    auto fixture = make_box_fixture({0, 0, 0}, {3, 1, 1}, {3, 1, 1});
    auto tags = std::make_shared<hellofem::mesh::MeshTags<int>>(
        fixture.mesh->topology(), 3, std::vector<std::int32_t> {1, 2},
        std::vector<int> {2, 1}, "sparse cells");

    // A temperature field that varies along x: T = x.
    HeatTransferSolver ht(fixture.mesh, fixture.boundary, fixture.cells, 1);
    auto& field = *ht.solution();
    const auto coords = ht.solution()->function_space()->tabulate_dof_coordinates(false);
    for (std::int32_t d = 0; d < ht.solution()->function_space()->dofmap()->index_map->size_local(); ++d)
        field.x()->array()[static_cast<std::size_t>(d)] = coords[3 * d];

    DomainProperty property(fixture.mesh, tags, {});
    property.bind_field("T", ht.solution());
    property.set_expression(2, "7.0"); // domain 2: constant
    property.set_expression(1, "2*T"); // domain 1: 2x
    REQUIRE(property.field_dependent());
    property.update(0.0);

    const auto& values = property.function()->x()->array();
    const auto& dofmap = *property.function()->function_space()->dofmap();
    auto value_on_cell = [&](std::int32_t cell) {
        return values[static_cast<std::size_t>(dofmap.cell_dofs(cell).front())];
    };
    // Domain 2 keeps its constant.
    REQUIRE(value_on_cell(1) == Approx(7.0));
    // Domain 1 evaluates 2*T at the cell centroid (cell 2 spans x in [2,3]).
    REQUIRE(value_on_cell(2) == Approx(5.0));
    // Untagged cell: untouched.
    REQUIRE(value_on_cell(0) == Approx(0.0));
}

TEST_CASE("DomainProperty: a law reading no field does not evaluate a bound one",
    "[app][physics]")
{
    // A physics publishes its solution for every material law that may read
    // it. A law that reads none must not pay for evaluating it — and must
    // not be refused a field whose value shape it never asked for.
    auto f = make_box_fixture({0, 0, 0}, {1, 1, 1}, {2, 2, 1});
    SolidMechanicsSolver sm(f.mesh, f.boundary, f.cells, 1); // vector solution
    DomainProperty property(f.mesh, f.cells, {});
    property.bind_field("u", sm.solution());
    property.set_expression(1, "3.0");
    REQUIRE_FALSE(property.field_dependent());

    property.update(0.0);

    const auto& dofmap = *property.function()->function_space()->dofmap();
    REQUIRE(property.function()->x()->array()[static_cast<std::size_t>(
                dofmap.cell_dofs(0).front())]
        == Approx(3.0));
}

TEST_CASE("FacetProperty: the value is carried by the cells around its boundary",
    "[app][physics]")
{
    // A facet integral reads its coefficient off the cell next to its facet,
    // so the value of a boundary lives on the cells around it — and only
    // there: a cell away from the boundary keeps zero, and no other
    // boundary's cells are touched.
    auto f = make_box_fixture({0, 0, 0}, {1, 1, 1}, {2, 2, 2});
    auto property = constant_facet_property(f.mesh, f.boundary, 6, 3.0);

    const std::size_t nc = f.mesh->topology()->index_map(3)->size_local();
    std::vector<char> around(nc, 0);
    std::size_t count = 0;
    const auto& indices = f.boundary->indices();
    const auto& boundaries = f.boundary->values();
    auto f_to_c = f.mesh->topology()->connectivity(2, 3);
    for (std::size_t i = 0; i < indices.size(); ++i) {
        if (boundaries[i] != 6)
            continue;
        const auto cell = f_to_c->links(indices[i])[0];
        if (not around[static_cast<std::size_t>(cell)]) {
            around[static_cast<std::size_t>(cell)] = 1;
            ++count;
        }
    }
    REQUIRE(count > 0);
    REQUIRE(count < nc); // the fixture separates the two sets

    const auto& values = property->function()->x()->array();
    const auto& dofmap = *property->function()->function_space()->dofmap();
    for (std::int32_t c = 0; c < static_cast<std::int32_t>(nc); ++c)
        REQUIRE(values[static_cast<std::size_t>(dofmap.cell_dofs(c).front())]
            == Approx(around[static_cast<std::size_t>(c)] ? 3.0 : 0.0));
}

TEST_CASE("FacetProperty: a value that varies needs one value per facet",
    "[app][physics]")
{
    // The value of a boundary is one value per facet, and the DG0 coefficient
    // a facet integral reads carries one value per cell: a cell the boundary
    // reaches through two facets cannot carry both. The fixture tags two
    // facets of one cell as one boundary — a model the app refuses rather
    // than solve with one facet's value standing in for the other's.
    auto f = make_box_fixture({0, 0, 0}, {1, 1, 1}, {2, 2, 2});
    auto f_to_c = f.mesh->topology()->connectivity(2, 3);
    std::vector<std::int32_t> indices;
    std::vector<int> values;
    std::int32_t first = -1;
    for (std::int32_t facet = 0;
        facet < static_cast<std::int32_t>(f_to_c->num_nodes()); ++facet) {
        auto cells = f_to_c->links(facet);
        if (cells.size() != 1)
            continue; // interior
        if (first < 0 or cells[0] == first) {
            first = cells[0];
            indices.push_back(facet);
            values.push_back(9);
        }
        if (indices.size() == 2)
            break;
    }
    REQUIRE(indices.size() == 2);
    auto tags = std::make_shared<hellofem::mesh::MeshTags<int>>(
        f.mesh->topology(), 2, std::move(indices), std::move(values), "doubled");

    auto varying = std::make_shared<FacetProperty>(f.mesh, tags, 9,
        std::unordered_map<std::string, double> {});
    varying->set_expression("1+x+y+z");
    REQUIRE_THROWS(varying->update(0.0));

    // One value everywhere is what the representation carries.
    auto constant = std::make_shared<FacetProperty>(f.mesh, tags, 9,
        std::unordered_map<std::string, double> {});
    constant->set_expression("5");
    constant->update(0.0);
    const auto& dofmap = *constant->function()->function_space()->dofmap();
    REQUIRE(constant->function()->x()->array()[static_cast<std::size_t>(
                dofmap.cell_dofs(first).front())]
        == Approx(5.0));
}

TEST_CASE("Electrostatics: -div(sigma grad V)=0 with V=V0 on x+, V=0 on x-", "[app][physics]")
{
    // 1x1x1 box, single layer in z. V solves Laplace; with V=1 on x+, 0 on x-,
    // the solution is V = x (linear), sigma = 1. A linear potential is in
    // every Lagrange space, so the discrete solution must reproduce it
    // exactly at any order — a patch test of the space, the assembly and the
    // boundary lift together.
    for (const int order : {1, 2, 3, 4}) {
        auto f = make_box_fixture({0, 0, 0}, {1, 1, 1}, {4, 4, 1});
        ElectrostaticsSolver es(f.mesh, f.boundary, f.cells, order);
        auto sigma = constant_property(f.mesh, f.cells, 1.0);
        es.set_conductivity(sigma);
        es.add_voltage_bc(2, ScalarExpression(1.0)); // x+ : V=1
        es.add_voltage_bc(1, ScalarExpression(0.0)); // x- : V=0

        es.solve_steady(0.0);
        auto V = es.solution();

        // V = x at every dof coordinate.
        auto coords = es.solution()->function_space()->tabulate_dof_coordinates(false);
        double max_err = 0;
        double v_max = -1e9, v_min = 1e9;
        for (std::int32_t d = 0; d < es.solution()->function_space()->dofmap()->index_map->size_local(); ++d) {
            const double x = coords[3 * d];
            const double val = V->x()->array()[static_cast<std::size_t>(d)];
            max_err = std::max(max_err, std::abs(val - x));
            v_max = std::max(v_max, val);
            v_min = std::min(v_min, val);
        }
        INFO("order " << order << ", max error = " << max_err << ", V range ["
                      << v_min << "," << v_max << "]");
        REQUIRE(max_err < 1e-10);
        REQUIRE(v_max == Catch::Approx(1.0).margin(1e-9));
        REQUIRE(v_min == Catch::Approx(0.0).margin(1e-9));
    }
}

TEST_CASE("Electrostatics: a linear potential is exact on a tetrahedral mesh",
    "[app][physics]")
{
    // The same patch test on tetrahedra, where a P3 and higher space needs
    // the element's dof permutation: a cube of six tetrahedra, V = z driven
    // by V=0 on z- and V=1 on z+, every other facet insulated.
    using hellofem::mesh::CellType;
    const std::vector<double> x {0, 0, 0, /**/ 1, 0, 0, /**/ 0, 1, 0,
        /**/ 1, 1, 0, /**/ 0, 0, 1, /**/ 1, 0, 1, /**/ 0, 1, 1, /**/ 1, 1, 1};
    // Kuhn's six tetrahedra around the main diagonal 0-7.
    const std::vector<std::int64_t> cells {0, 1, 3, 7, /**/ 0, 1, 5, 7,
        /**/ 0, 2, 3, 7, /**/ 0, 2, 6, 7, /**/ 0, 4, 5, 7, /**/ 0, 4, 6, 7};
    auto mesh = std::make_shared<hellofem::mesh::Mesh<double>>(
        hellofem::mesh::create_mesh(std::span<const std::int64_t>(cells),
            CellType::tetrahedron, x, 3));
    auto boundary = hellofem::app::test::boundary_tags(*mesh);
    const std::size_t nc = mesh->topology()->index_map(3)->size_local();
    std::vector<std::int32_t> cell_idx(nc);
    std::iota(cell_idx.begin(), cell_idx.end(), 0);
    auto cell_tags = std::make_shared<hellofem::mesh::MeshTags<int>>(
        mesh->topology(), 3, std::move(cell_idx),
        std::vector<int>(nc, 1), "cells");

    for (const int order : {1, 2, 3, 4}) {
        ElectrostaticsSolver es(mesh, boundary, cell_tags, order);
        es.set_conductivity(constant_property(mesh, cell_tags, 1.0));
        es.add_voltage_bc(6, ScalarExpression(1.0)); // z+ : V=1
        es.add_voltage_bc(5, ScalarExpression(0.0)); // z- : V=0
        es.solve_steady(0.0);

        auto coords = es.solution()->function_space()->tabulate_dof_coordinates(false);
        double max_err = 0;
        for (std::int32_t d = 0;
            d < es.solution()->function_space()->dofmap()->index_map->size_local(); ++d)
            max_err = std::max(max_err,
                std::abs(es.solution()->x()->array()[static_cast<std::size_t>(d)]
                    - coords[3 * d + 2]));
        INFO("order " << order << ", max error = " << max_err);
        REQUIRE(max_err < 1e-10);
    }
}

TEST_CASE("Electrostatics: voltage-dependent conductivity drives a nonlinear solve",
    "[app][physics]")
{
    // sigma = 1 + V makes -div(sigma grad V) = 0 nonlinear: (1 + V) V' = C
    // with V(0) = 0, V(1) = 1 gives V = sqrt(1 + 3x) - 1, whereas a single
    // linear solve would return the linear profile x. The distance to that
    // profile is what tells a solved nonlinearity from a skipped one.
    auto f = make_box_fixture({0, 0, 0}, {1, 1, 1}, {8, 1, 1});
    ElectrostaticsSolver es(f.mesh, f.boundary, f.cells, 1);
    es.add_voltage_bc(2, ScalarExpression(1.0));
    es.add_voltage_bc(1, ScalarExpression(0.0));
    auto sigma = std::make_shared<DomainProperty>(f.mesh, f.cells,
        std::unordered_map<std::string, double> {});
    sigma->bind_field("V", es.solution());
    sigma->set_expression(1, "1 + V");
    es.set_conductivity(sigma);

    es.solve_steady(0.0);

    auto coords = es.solution()->function_space()->tabulate_dof_coordinates(false);
    double err_exact = 0, err_linear = 0;
    for (std::int32_t d = 0; d < es.solution()->function_space()->dofmap()->index_map->size_local(); ++d) {
        const double x = coords[3 * d];
        const double value
            = es.solution()->x()->array()[static_cast<std::size_t>(d)];
        err_exact = std::max(err_exact,
            std::abs(value - (std::sqrt(1.0 + 3.0 * x) - 1.0)));
        err_linear = std::max(err_linear, std::abs(value - x));
    }
    INFO("nonlinear Laplace: err(exact) = " << err_exact
                                            << ", err(linear profile) = " << err_linear);
    REQUIRE(err_exact < 1e-7);
    REQUIRE(err_linear > 0.05);
}

TEST_CASE("HeatTransfer: steady -div(k grad T)=0 with Robin convection", "[app][physics]")
{
    // 1D rod along x in a 1x1x1 box (single y,z layer): -d/dx(k dT/dx)=0,
    // T=1 on x-, Robin h(T-Tinf)=h*T on x+ (Tinf=0). With k=1, h=1:
    // T = a + b·x, T(0)=1 => a=1; Robin: -k dT/dx = h·T(1) => -b = 1+b
    // => b = -1/2. Hence T(x) = 1 - x/2, T(1)=0.5.
    auto f = make_box_fixture({0, 0, 0}, {1, 1, 1}, {4, 4, 1});
    HeatTransferSolver ht(f.mesh, f.boundary, f.cells, 1);
    auto k = constant_property(f.mesh, f.cells, 1.0);
    ht.set_conductivity(k);
    ht.add_temperature_bc(1, ScalarExpression(1.0)); // x- : T=1
    ht.add_convection(constant_facet_property(f.mesh, f.boundary, 2, 1.0),
        constant_facet_property(f.mesh, f.boundary, 2, 0.0)); // x+ : h=1, Tinf=0

    ht.solve_steady(0.0);
    auto T = ht.solution();

    auto coords = ht.solution()->function_space()->tabulate_dof_coordinates(false);
    double max_err = 0;
    for (std::int32_t d = 0; d < ht.solution()->function_space()->dofmap()->index_map->size_local(); ++d) {
        const double x = coords[3 * d];
        const double exact = 1.0 - 0.5 * x;
        max_err = std::max(max_err,
            std::abs(T->x()->array()[static_cast<std::size_t>(d)] - exact));
    }
    INFO("heat max error = " << max_err);
    REQUIRE(max_err < 1e-8);
}

TEST_CASE("HeatTransfer: a boundary value is read on the boundary", "[app][physics]")
{
    // The Robin coefficient of x+ is h = 1 + x, which is 2 *on the boundary*:
    // with k = 1 and T = 1 on x-, the exact end temperature is
    // T(1) = 1 - h(1)/(1 + h(1)) = 1/3. Reading h at the centroid of the cells
    // next to the boundary instead — x = 0.875 on this four-cell rod — gives
    // 0.347826, which is what a value carried by the cells rather than by the
    // boundary returns.
    auto f = make_box_fixture({0, 0, 0}, {1, 1, 1}, {4, 4, 1});
    HeatTransferSolver ht(f.mesh, f.boundary, f.cells, 1);
    ht.set_conductivity(constant_property(f.mesh, f.cells, 1.0));
    ht.add_temperature_bc(1, ScalarExpression(1.0)); // x- : T=1
    auto h = std::make_shared<FacetProperty>(f.mesh, f.boundary, 2,
        std::unordered_map<std::string, double> {});
    h->set_expression("1+x");
    // A condition carries the data of one boundary: coefficients of another
    // boundary are refused rather than left out of the integral.
    REQUIRE_THROWS(ht.add_convection(
        h, constant_facet_property(f.mesh, f.boundary, 1, 0.0)));
    ht.add_convection(h, constant_facet_property(f.mesh, f.boundary, 2, 0.0));

    ht.solve_steady(0.0);

    auto coords = ht.solution()->function_space()->tabulate_dof_coordinates(false);
    double at_end = 0.0;
    for (std::int32_t d = 0; d < ht.solution()->function_space()->dofmap()->index_map->size_local(); ++d)
        if (coords[3 * d] > 0.99)
            at_end = ht.solution()->x()->array()[static_cast<std::size_t>(d)];
    INFO("T(1) = " << at_end);
    REQUIRE(at_end == Approx(1.0 / 3.0).margin(1e-12));
}

TEST_CASE("HeatTransfer: nonlinear k(T) matches the analytic steady profile", "[app][physics]")
{
    // Rod along x: -d/dx(k(T) dT/dx) = 0 with k = k0 (1 + a (T - Tref)).
    // The flux is constant, so with u(T) = (T-Tref) + a (T-Tref)^2 / 2 the
    // exact profile satisfies u(T(x)) = u(T0) + (u(T1) - u(T0)) x / L.
    const double k0 = 45.0, a = -0.001, t_ref = 293.15;
    const double t0 = 293.15, t1 = 373.15, length = 1.0;
    auto u = [&](double T) {
        return (T - t_ref) + 0.5 * a * (T - t_ref) * (T - t_ref);
    };
    auto exact = [&](double x) {
        const double target = u(t0) + (u(t1) - u(t0)) * x / length;
        // Solve (T - Tref) + a/2 (T - Tref)^2 = target for T > Tref.
        const double b = 1.0, c = -target;
        return t_ref + (-b + std::sqrt(b * b - 2.0 * a * c)) / a;
    };

    auto f = make_box_fixture({0, 0, 0}, {length, 0.2, 0.2}, {8, 1, 1});
    HeatTransferSolver ht(f.mesh, f.boundary, f.cells, 1);
    auto k = std::make_shared<DomainProperty>(f.mesh, f.cells,
        std::unordered_map<std::string, double> {
            {"k0", k0}, {"alpha_k", a}, {"Tref", t_ref}});
    k->bind_field("T", ht.solution());
    k->set_expression(1, "k0*(1+alpha_k*(T-Tref))");
    ht.set_conductivity(k);
    ht.add_temperature_bc(1, ScalarExpression(t0));
    ht.add_temperature_bc(2, ScalarExpression(t1));

    ht.solve_steady(0.0);

    auto coords = ht.solution()->function_space()->tabulate_dof_coordinates(false);
    double max_err = 0;
    for (std::int32_t d = 0; d < ht.solution()->function_space()->dofmap()->index_map->size_local(); ++d) {
        const double x = coords[3 * d];
        max_err = std::max(max_err,
            std::abs(ht.solution()->x()->array()[static_cast<std::size_t>(d)]
                - exact(x)));
    }
    INFO("nonlinear k(T) max error = " << max_err);
    REQUIRE(max_err < 1e-3);
}

namespace {

    /// 1-based boundary id on the interior facets of the plane `x = value`.
    /// The facets of the box fixtures are tagged 1..6 for its six sides, so a
    /// tag above those is free for an imprinted face.
    std::shared_ptr<hellofem::mesh::MeshTags<int>> internal_plane_tags(
        const hellofem::mesh::Mesh<double>& mesh, double value, int id)
    {
        const int tdim = mesh.topology()->dim();
        auto f_to_c = mesh.topology()->connectivity(tdim - 1, tdim);
        auto f_to_v = mesh.topology()->connectivity(tdim - 1, 0);
        const auto [vc, shape] = hellofem::mesh::compute_vertex_coords(mesh);
        const std::size_t nv = shape[1];

        std::vector<std::int32_t> indices;
        for (std::int32_t f = 0; f < static_cast<std::int32_t>(f_to_v->num_nodes()); ++f) {
            if (f_to_c->num_links(f) != 2)
                continue; // exterior
            bool on_plane = true;
            for (auto v : f_to_v->links(f))
                on_plane = on_plane and vc[static_cast<std::size_t>(v)] == value;
            if (on_plane)
                indices.push_back(f);
        }
        REQUIRE_FALSE(indices.empty());
        return std::make_shared<hellofem::mesh::MeshTags<int>>(mesh.topology(),
            tdim - 1, std::move(indices), std::vector<int>(1, id), "internal");
    }

    /// The layer operator alone: the heat field with no conductivity, no
    /// source and no condition but the thin layer, so the assembled matrix
    /// holds nothing else.
    la::MatrixCSR<double> thin_layer_operator(
        const hellofem::app::test::BoxFixture& f,
        const std::shared_ptr<hellofem::mesh::MeshTags<int>>& boundary,
        int layer_boundary, double ds, double k, int order)
    {
        HeatTransferSolver ht(f.mesh, boundary, f.cells, order);
        ht.set_conductivity(constant_property(f.mesh, f.cells, 0.0));
        ht.add_source({1}, constant_property(f.mesh, f.cells, 0.0));
        ht.add_thin_layer(
            constant_facet_property(f.mesh, boundary, layer_boundary, ds),
            constant_facet_property(f.mesh, boundary, layer_boundary, k));
        ht.refresh(0.0);
        la::MatrixCSR<double> A(ht.pattern());
        la::Vector<double> b(ht.solution()->x()->index_map(),
            ht.solution()->x()->bs());
        ht.assemble_steady(A, b, 0.0);
        return A;
    }

} // namespace

TEST_CASE("HeatTransfer: a thin layer conducts along the boundary only",
    "[app][physics]")
{
    // The layer's energy for a linear temperature is
    //     v^T A v = int_Gamma ds k |grad_t T|^2 dA
    // with grad_t the gradient projected onto the boundary — the layer does
    // not conduct across its thickness. A temperature with a normal component
    // is what tells the projection from its absence: with T = a + bx x +
    // by y + bz z on the z+ face of the unit box, the layer's energy is
    // ds k (bx^2 + by^2), against ds k (bx^2 + by^2 + bz^2) for the full
    // gradient. Both the domain's conductivity and its source are zero here,
    // so the matrix holds the layer and nothing else.
    const double ds = 1e-4, k_layer = 400.0;
    const double bx = 1.0, by = -2.0, bz = 3.0;
    auto f = make_box_fixture({0, 0, 0}, {1, 1, 1}, {2, 2, 2});

    for (const int order : {1, 2}) {
        auto A = thin_layer_operator(f, f.boundary, 6, ds, k_layer, order);
        HeatTransferSolver ht(f.mesh, f.boundary, f.cells, order);
        const auto coords = ht.solution()->function_space()->tabulate_dof_coordinates(false);
        la::Vector<double> v(ht.solution()->x()->index_map(),
            ht.solution()->x()->bs());
        for (std::int32_t d = 0;
            d < ht.solution()->function_space()->dofmap()->index_map->size_local(); ++d)
            v.array()[static_cast<std::size_t>(d)]
                = 1.0 + bx * coords[3 * d] + by * coords[3 * d + 1]
                + bz * coords[3 * d + 2];
        la::Vector<double> Av(v.index_map(), v.bs());
        A.mult(v, Av);
        double energy = 0.0;
        for (std::size_t i = 0; i < v.array().size(); ++i)
            energy += v.array()[i] * Av.array()[i];

        const double tangential = ds * k_layer * (bx * bx + by * by);
        const double full = ds * k_layer * (bx * bx + by * by + bz * bz);
        INFO("order " << order << ": layer energy " << energy
                      << ", tangential " << tangential << ", full " << full);
        REQUIRE(energy == Approx(tangential).epsilon(1e-12));
    }
}

TEST_CASE("HeatTransfer: a thin layer on an imprinted face conducts too",
    "[app][physics]")
{
    // The interconnect of a package is a copper layer imprinted *inside* a
    // domain: the boundary is an interior facet, shared by two cells. Its
    // dofs are the same for both cells, so the layer's operator must be the
    // same one as on an exterior boundary of the same area.
    const double ds = 5e-6, k_layer = 400.0;
    const double bx = 1.0, by = -2.0, bz = 3.0;
    auto f = make_box_fixture({0, 0, 0}, {1, 1, 1}, {4, 1, 1});
    auto tags = internal_plane_tags(*f.mesh, 0.25, 7);

    for (const int order : {1, 2}) {
        auto A = thin_layer_operator(f, tags, 7, ds, k_layer, order);
        HeatTransferSolver ht(f.mesh, tags, f.cells, order);
        const auto coords = ht.solution()->function_space()->tabulate_dof_coordinates(false);
        la::Vector<double> v(ht.solution()->x()->index_map(),
            ht.solution()->x()->bs());
        for (std::int32_t d = 0;
            d < ht.solution()->function_space()->dofmap()->index_map->size_local(); ++d)
            v.array()[static_cast<std::size_t>(d)]
                = 1.0 + bx * coords[3 * d] + by * coords[3 * d + 1]
                + bz * coords[3 * d + 2];
        la::Vector<double> Av(v.index_map(), v.bs());
        A.mult(v, Av);
        double energy = 0.0;
        for (std::size_t i = 0; i < v.array().size(); ++i)
            energy += v.array()[i] * Av.array()[i];

        // The imprinted face spans y and z, so its normal is x and the
        // tangential gradient keeps by and bz.
        const double tangential = ds * k_layer * (by * by + bz * bz);
        INFO("order " << order << ": layer energy " << energy
                      << ", tangential " << tangential);
        REQUIRE(energy == Approx(tangential).epsilon(1e-12));
    }
}

TEST_CASE("HeatTransfer: a thin layer does not conduct across its thickness",
    "[app][physics]")
{
    // A bar along x, a uniform source, T = 0 at x- and its far end insulated:
    // the exact profile is T(x) = Q (L x - x^2/2) / k, whatever the length.
    // A layer on the *far* face — the one whose normal is the direction the
    // temperature varies in — conducts only along the boundary, and its
    // tangential gradient is the part of grad T the face does not carry,
    // which here is nothing. So the layer leaves the profile exactly where it
    // was. A projection that kept the normal component instead would add the
    // layer's conductance to the bar and land on Q/(2 (k + ds k_layer)) —
    // 1/22 against 1/2 for the numbers below.
    const double k = 1.0, ds = 1e-3, k_layer = 1e4, Q = 1.0;
    auto f = make_box_fixture({0, 0, 0}, {1, 1, 1}, {4, 1, 1});

    for (const bool with_layer : {false, true}) {
        HeatTransferSolver ht(f.mesh, f.boundary, f.cells, 2);
        ht.set_conductivity(constant_property(f.mesh, f.cells, k));
        ht.add_source({1}, constant_property(f.mesh, f.cells, Q));
        ht.add_temperature_bc(1, ScalarExpression(0.0)); // x- : T = 0
        if (with_layer)
            ht.add_thin_layer(
                constant_facet_property(f.mesh, f.boundary, 2, ds),
                constant_facet_property(f.mesh, f.boundary, 2, k_layer));
        ht.solve_steady(0.0);

        const auto coords = ht.solution()->function_space()->tabulate_dof_coordinates(false);
        double err = 0.0, at_one = -1e9;
        for (std::int32_t d = 0;
            d < ht.solution()->function_space()->dofmap()->index_map->size_local(); ++d) {
            const double x = coords[3 * d];
            const double value
                = ht.solution()->x()->array()[static_cast<std::size_t>(d)];
            err = std::max(err,
                std::abs(value - Q * (x - 0.5 * x * x) / k));
            if (std::abs(x - 1.0) < 1e-9)
                at_one = std::max(at_one, value);
        }
        INFO("layer " << with_layer << ": max error " << err
                      << ", T(1) = " << at_one);
        REQUIRE(err < 1e-12);
        REQUIRE(at_one == Approx(0.5 * Q / k).margin(1e-12));
    }
}

TEST_CASE("HeatTransfer: a thin layer takes the material of its own boundary",
    "[app][physics]")
{
    // Two boundaries of one domain carry two different surface materials,
    // and one thin layer feature selects both. Each boundary's layer must
    // use its own material's conductivity: a facet integral reads its
    // coefficient off the adjacent cell, so the selection is assembled as
    // one integral per material, each carrying one value. Reading the
    // boundaries' materials into the domain's cells instead gives both
    // integrals the same conductivity, which is a different field. The
    // volumetric source keeps the bulk field off a linear profile, which a
    // layer parallel to the flux would leave untouched.
    auto f = make_box_fixture({0, 0, 0}, {1, 1, 1}, {2, 2, 2});
    const double thickness = 0.1, k_lower = 1.0, k_upper = 2.0;

    LoadedMesh lm;
    lm.mesh = f.mesh;
    lm.order = 1;
    lm.cell_tags = f.cells;
    lm.facet_tags = f.boundary;
    lm.num_domains = 1;
    lm.num_boundaries = 6;

    ModelScript model;
    model.materials.push_back(Material {"bulk", {1}, {}, 3,
        {{"thermalconductivity", "1"}}});
    model.materials.push_back(Material {"matA", {}, {5}, 2,
        {{"thermalconductivity", "1"}}});
    model.materials.push_back(Material {"matB", {}, {6}, 2,
        {{"thermalconductivity", "2"}}});
    Physics physics;
    physics.tag = "ht";
    physics.type = "HeatTransfer";
    physics.features.push_back({"init1", "Init", {}, {{"Tinit", "0"}}});
    physics.features.push_back({"hs1", "HeatSource", {1}, {{"Q0", "1"}}});
    physics.features.push_back(
        {"temp1", "TemperatureBoundary", {1}, {{"T0", "0"}}});
    physics.features.push_back({"sls1", "SolidLayeredShell", {5, 6},
        {{"UserDefThicknessLayerType", "Conductive"}, {"lth_mat", "userdef"},
            {"lth", "0.1"}, {"k_mat", "from_mat"}}});
    model.physics.push_back(physics);

    CaseContext ctx(model, lm, TimeSettings {});
    // The order the app resolves for the physics is the one the reference
    // has to be built with; a different one is a different discretization.
    const int order = ctx.element_order(physics, "temperature");

    // Reference: the same problem with its two layers, each carrying the
    // conductivity of the boundary it sits on, added one by one.
    HeatTransferSolver reference(f.mesh, f.boundary, f.cells, order);
    reference.set_conductivity(constant_property(f.mesh, f.cells, 1.0));
    reference.add_source({1}, constant_property(f.mesh, f.cells, 1.0));
    reference.add_temperature_bc(1, ScalarExpression(0.0));
    reference.add_thin_layer(
        constant_facet_property(f.mesh, f.boundary, 5, thickness),
        constant_facet_property(f.mesh, f.boundary, 5, k_lower));
    reference.add_thin_layer(
        constant_facet_property(f.mesh, f.boundary, 6, thickness),
        constant_facet_property(f.mesh, f.boundary, 6, k_upper));
    reference.refresh(0.0);
    reference.solve_steady(0.0);

    // The app's own binding of the model.
    const FieldKind* kind = field_kind("HeatTransfer");
    REQUIRE(kind != nullptr);
    auto bound = kind->create(physics, ctx);
    bound->initialize(0.0);
    bound->solve_level(0.0);

    // Compare where the model's own export reads the field: at the mesh
    // vertices, through the variable the physics publishes.
    const auto [vc, vshape] = hellofem::mesh::compute_vertex_coords(*f.mesh);
    const std::size_t nv = vshape[1];
    std::vector<double> points(nv * 3);
    for (std::size_t i = 0; i < nv; ++i)
        for (int q = 0; q < 3; ++q)
            points[i * 3 + static_cast<std::size_t>(q)] = vc[q * nv + i];

    auto c_to_v = f.mesh->topology()->connectivity(3, 0);
    std::vector<std::int32_t> cells(nv, -1);
    for (std::int32_t c = 0; c < static_cast<std::int32_t>(c_to_v->num_nodes()); ++c)
        for (auto v : c_to_v->links(c))
            if (cells[static_cast<std::size_t>(v)] < 0)
                cells[static_cast<std::size_t>(v)] = c;

    auto read = [&](const Variable& variable) {
        std::vector<double> values(nv);
        variable.eval(points, cells, values);
        return values;
    };
    const auto& variables = bound->variables();
    REQUIRE(variables.size() == 1);
    const auto actual = read(variables.front());
    auto worst_from = [&](const std::vector<double>& other) {
        double worst = 0.0;
        for (std::size_t i = 0; i < nv; ++i)
            worst = std::max(worst, std::abs(actual[i] - other[i]));
        return worst;
    };
    const auto expected = read(scalar_variable("T", "(K)", reference.solution()));
    const double against_reference = worst_from(expected);
    INFO("worst difference from the per-boundary reference = "
        << against_reference);
    REQUIRE(against_reference < 1e-9);

    // The discrimination: one conductivity for both layers — the single
    // value a domain-wide coefficient can carry — is a different field,
    // and a macroscopic one, so the agreement above is not vacuous.
    HeatTransferSolver single(f.mesh, f.boundary, f.cells, order);
    single.set_conductivity(constant_property(f.mesh, f.cells, 1.0));
    single.add_source({1}, constant_property(f.mesh, f.cells, 1.0));
    single.add_temperature_bc(1, ScalarExpression(0.0));
    single.add_thin_layer(
        constant_facet_property(f.mesh, f.boundary, 5, thickness),
        constant_facet_property(f.mesh, f.boundary, 5, k_lower));
    single.add_thin_layer(
        constant_facet_property(f.mesh, f.boundary, 6, thickness),
        constant_facet_property(f.mesh, f.boundary, 6, k_lower));
    single.refresh(0.0);
    single.solve_steady(0.0);
    const double against_single = worst_from(
        read(scalar_variable("T", "(K)", single.solution())));
    INFO("worst difference from the single-conductivity assembly = "
        << against_single);
    REQUIRE(against_single > 1e-3);
}

TEST_CASE("HeatTransfer: a source is integrated over the domains it belongs to",
    "[app][physics]")
{
    // One box, two domains, and a source on each with its own strength: the
    // model selects domain 1 for Q = 2 and domain 2 for Q = 1, so the load is
    // two integrals over disjoint cells. A binding that gave a feature's
    // source the whole mesh, or the other domain, is a different field.
    auto f = make_box_fixture({0, 0, 0}, {1, 1, 1}, {2, 2, 2});
    const std::size_t nc = f.mesh->topology()->index_map(3)->size_local();
    REQUIRE(nc > 1);

    // Two domains: the lower half of the cells, and the upper half.
    std::vector<std::int32_t> indices(nc);
    std::iota(indices.begin(), indices.end(), 0);
    std::vector<int> domains(nc);
    for (std::size_t c = 0; c < nc; ++c)
        domains[c] = c < nc / 2 ? 1 : 2;
    auto cells = std::make_shared<hellofem::mesh::MeshTags<int>>(
        f.mesh->topology(), 3, std::move(indices), std::move(domains), "cells");

    LoadedMesh lm;
    lm.mesh = f.mesh;
    lm.order = 1;
    lm.cell_tags = cells;
    lm.facet_tags = f.boundary;
    lm.num_domains = 2;
    lm.num_boundaries = 6;

    ModelScript model;
    model.materials.push_back(Material {"mat1", {1, 2}, {}, 3,
        {{"thermalconductivity", "1"}}});
    Physics physics;
    physics.tag = "ht";
    physics.type = "HeatTransfer";
    physics.features.push_back({"init1", "Init", {}, {{"Tinit", "0"}}});
    physics.features.push_back({"hs1", "HeatSource", {1}, {{"Q0", "2"}}});
    physics.features.push_back({"hs2", "HeatSource", {2}, {{"Q0", "1"}}});
    physics.features.push_back(
        {"temp1", "TemperatureBoundary", {1}, {{"T0", "0"}}});
    model.physics.push_back(physics);

    CaseContext ctx(model, lm, TimeSettings {});
    const int order = ctx.element_order(physics, "temperature");

    // A DG0 property holding a constant on both domains.
    const auto uniform = [&](double value) {
        auto property = std::make_shared<DomainProperty>(f.mesh, cells,
            std::unordered_map<std::string, double> {});
        property->set_expression(1, std::to_string(value));
        property->set_expression(2, std::to_string(value));
        property->update(0.0);
        return property;
    };

    // Reference: the same two sources, each registered on its own domain.
    HeatTransferSolver reference(f.mesh, f.boundary, cells, order);
    reference.set_conductivity(uniform(1.0));
    reference.add_source({1}, uniform(2.0));
    reference.add_source({2}, uniform(1.0));
    reference.add_temperature_bc(1, ScalarExpression(0.0));
    reference.refresh(0.0);
    reference.solve_steady(0.0);

    const FieldKind* kind = field_kind("HeatTransfer");
    REQUIRE(kind != nullptr);
    auto bound = kind->create(physics, ctx);
    bound->initialize(0.0);
    bound->solve_level(0.0);

    const auto [vc, vshape] = hellofem::mesh::compute_vertex_coords(*f.mesh);
    const std::size_t nv = vshape[1];
    std::vector<double> points(nv * 3);
    for (std::size_t i = 0; i < nv; ++i)
        for (int q = 0; q < 3; ++q)
            points[i * 3 + static_cast<std::size_t>(q)] = vc[q * nv + i];
    auto c_to_v = f.mesh->topology()->connectivity(3, 0);
    std::vector<std::int32_t> vertex_cells(nv, -1);
    for (std::int32_t c = 0; c < static_cast<std::int32_t>(c_to_v->num_nodes()); ++c)
        for (auto v : c_to_v->links(c))
            if (vertex_cells[static_cast<std::size_t>(v)] < 0)
                vertex_cells[static_cast<std::size_t>(v)] = c;

    auto read = [&](const Variable& variable) {
        std::vector<double> values(nv);
        variable.eval(points, vertex_cells, values);
        return values;
    };
    const auto& variables = bound->variables();
    REQUIRE(variables.size() == 1);
    const auto actual = read(variables.front());
    auto worst_from = [&](const std::vector<double>& other) {
        double worst = 0.0;
        for (std::size_t i = 0; i < nv; ++i)
            worst = std::max(worst, std::abs(actual[i] - other[i]));
        return worst;
    };
    const auto expected = read(scalar_variable("T", "(K)", reference.solution()));
    const double against_reference = worst_from(expected);
    INFO("worst difference from the per-domain reference = " << against_reference);
    REQUIRE(against_reference < 1e-9);

    // The discrimination: the two strengths swapped between the domains is a
    // different field, and a macroscopic one, so the agreement above is not
    // vacuous.
    HeatTransferSolver swapped(f.mesh, f.boundary, cells, order);
    swapped.set_conductivity(uniform(1.0));
    swapped.add_source({1}, uniform(1.0));
    swapped.add_source({2}, uniform(2.0));
    swapped.add_temperature_bc(1, ScalarExpression(0.0));
    swapped.refresh(0.0);
    swapped.solve_steady(0.0);
    const double against_swapped = worst_from(
        read(scalar_variable("T", "(K)", swapped.solution())));
    INFO("worst difference from the swapped assembly = " << against_swapped);
    REQUIRE(against_swapped > 1e-3);
}

TEST_CASE("HeatTransfer: two sources on one domain are refused", "[app][physics]")
{
    // COMSOL replaces a domain's source when a later feature selects that
    // domain again. The app integrates one source per feature, so two
    // features sharing a domain would add where the reference replaces: a
    // model it cannot solve is refused rather than solved as a different one.
    auto f = make_box_fixture({0, 0, 0}, {1, 1, 1}, {1, 1, 1});
    HeatTransferSolver ht(f.mesh, f.boundary, f.cells, 1);
    ht.add_source({1}, constant_property(f.mesh, f.cells, 1.0));
    REQUIRE_THROWS(ht.add_source({1, 2}, constant_property(f.mesh, f.cells, 1.0)));
    // A source on a domain no other source holds is what the app solves.
    ht.add_source({2}, constant_property(f.mesh, f.cells, 1.0));
}

TEST_CASE("SolidMechanics: blocked assembly — no load with Fixed gives u=0", "[app][physics]")
{
    auto f = make_box_fixture({0, 0, 0}, {1, 1, 1}, {2, 2, 1});
    SolidMechanicsSolver sm(f.mesh, f.boundary, f.cells, 1);
    auto E = constant_property(f.mesh, f.cells, 200e9);
    auto nu = constant_property(f.mesh, f.cells, 0.3);
    sm.set_elastic(E, nu);
    sm.add_fixed_bc(1); // x- face: u=v=w=0

    sm.solve_steady(0.0);
    auto u = sm.solution();
    double max_mag = 0;
    for (double v : u->x()->array())
        max_mag = std::max(max_mag, std::abs(v));
    INFO("no-load solid max |u| = " << max_mag);
    REQUIRE(max_mag < 1e-10);
}

TEST_CASE("SolidMechanics: uniform thermal expansion of a clamped bar", "[app][physics]")
{
    // Slender bar along x: uniform DT=100, alpha=1e-5 -> free expansion would
    // be u_x = alpha*DT*x = 1e-3 at x=1. Fixed (u=v=w=0) on the x- face adds a
    // lateral-constraint boundary layer, so u_x(1) stays near alpha*DT*L and
    // the displacement is smooth and monotone.
    auto f = make_box_fixture({0, 0, 0}, {1, 0.3, 0.3}, {10, 3, 3});
    SolidMechanicsSolver sm(f.mesh, f.boundary, f.cells, 1);
    auto E = constant_property(f.mesh, f.cells, 200e9);
    auto nu = constant_property(f.mesh, f.cells, 0.3);
    auto alpha = constant_property(f.mesh, f.cells, 1e-5);
    auto T = constant_property(f.mesh, f.cells, 393.15); // DT = 100 above Tref
    sm.set_elastic(E, nu);
    sm.set_thermal_expansion(T->function(), alpha, 293.15);
    sm.add_fixed_bc(1); // x- face clamped

    sm.solve_steady(0.0);
    auto u = sm.solution();
    const auto& xa = u->x()->array();

    // Blocked space: dof coordinates are per physical dof (bs*block + comp),
    // so vertex d lives at dofcoords[(3*d)*gdim + q] = coords[9*d+q], and the
    // solution component c of vertex d is xa[3*d+c].
    auto coords = sm.solution()->function_space()->tabulate_dof_coordinates(false);
    double ux_max = 0, ux_at_1 = -1e9;
    double max_lat = 0;
    for (std::int32_t d = 0; d < sm.solution()->function_space()->dofmap()->index_map->size_local(); ++d) {
        const std::size_t cd = static_cast<std::size_t>(9 * d);
        const double x = coords[cd];
        const double ux = xa[static_cast<std::size_t>(3 * d)];
        const double uy = xa[static_cast<std::size_t>(3 * d + 1)];
        const double uz = xa[static_cast<std::size_t>(3 * d + 2)];
        ux_max = std::max(ux_max, ux);
        if (std::abs(x - 1.0) < 1e-9)
            ux_at_1 = std::max(ux_at_1, ux);
        max_lat = std::max(max_lat, std::max(std::abs(uy), std::abs(uz)));
    }
    INFO("thermal bar ux_max = " << ux_max << ", ux(x=1)=" << ux_at_1
                                 << ", max lateral |u| = " << max_lat);
    REQUIRE(ux_max > 0.9e-3);
    REQUIRE(ux_max < 1.1e-3);
    REQUIRE(ux_at_1 > 0.9e-3); // end face essentially alpha*DT*L
    REQUIRE(max_lat < 2e-3); // bounded lateral deformation
}

TEST_CASE("SolidMechanics: a varying temperature's load is the exact integral",
    "[app][physics]")
{
    // The uniform-expansion test above only exercises the boundary part of the
    // thermal load. For a constant sigma_th, int grad phi_i is zero at every
    // interior node (the gradients sum to the gradient of one), so the whole
    // load sits on the boundary nodes and that test passes for any interior
    // quadrature of the load. A temperature that varies makes the load
    // interior, which is what this checks.
    //
    // For v = (x, y, z) the load's work is b.v = int sigma_th div v, and with
    // T = Tref + g x on a box [0, Lx] x [0, Ly] x [0, Lz] that is
    //     3 alpha g E/(1 - 2 nu) * V * xbar,  V = Lx Ly Lz, xbar = Lx/2,
    // since sigma_th = alpha (T - Tref) E/(1 - 2 nu). v is linear, so the
    // discrete work reproduces the integral exactly and the identity holds to
    // round-off rather than to a mesh error.
    const double Lx = 1.0, Ly = 0.5, Lz = 0.5;
    const double g = 100.0; // K/m
    const double E = 200e9, nu = 0.3, alpha = 1e-5, Tref = 293.15;

    auto f = make_box_fixture({0, 0, 0}, {Lx, Ly, Lz}, {4, 3, 3});
    SolidMechanicsSolver sm(f.mesh, f.boundary, f.cells, 1);
    sm.set_elastic(constant_property(f.mesh, f.cells, E),
        constant_property(f.mesh, f.cells, nu));

    // A scalar space carrying T = Tref + g x.
    HeatTransferSolver ht(f.mesh, f.boundary, f.cells, 1);
    auto T = std::make_shared<hellofem::fem::Function<double>>(ht.solution()->function_space());
    T->x()->set(0.0);
    auto Tcoords = ht.solution()->function_space()->tabulate_dof_coordinates(false);
    for (std::int32_t d = 0;
        d < ht.solution()->function_space()->dofmap()->index_map->size_local(); ++d)
        T->x()->array()[static_cast<std::size_t>(d)]
            = Tref + g * Tcoords[static_cast<std::size_t>(3 * d)];
    sm.set_thermal_expansion(T, constant_property(f.mesh, f.cells, alpha), Tref);
    sm.refresh(0.0);

    la::MatrixCSR<double> A(sm.pattern());
    la::Vector<double> b(sm.solution()->x()->index_map(),
        sm.solution()->x()->bs());
    sm.assemble_steady(A, b, 0.0);

    // The load against v = (x, y, z): physical dof i is component i % 3 of its
    // node, whose coordinate is at coords[3 * i + (i % 3)].
    auto coords = sm.solution()->function_space()->tabulate_dof_coordinates(false);
    double work = 0.0;
    for (std::size_t i = 0; i < b.array().size(); ++i)
        work += b.array()[i] * coords[3 * i + (i % 3)];

    const double exact = 3.0 * alpha * g * E / (1 - 2 * nu) * (Lx * Ly * Lz)
        * (Lx / 2);
    INFO("load work = " << work << " against the exact " << exact);
    REQUIRE(std::abs(work - exact) < 1e-8 * std::abs(exact));
}

TEST_CASE("SolidMechanics: AMG preconditioning keeps the block structure",
    "[app][physics]")
{
    // Elasticity is a vector operator: the three components of a node are
    // strongly coupled, so a hierarchy that coarsens the scalar expansion of
    // the matrix splits them over different aggregates and barely helps.
    // Both solves below must converge on the same displacement; the blocked
    // one must do it in a fraction of the iterations.
    auto f = make_box_fixture({0, 0, 0}, {1, 0.3, 0.05}, {32, 10, 3});
    SolidMechanicsSolver sm(f.mesh, f.boundary, f.cells, 1);
    auto E = constant_property(f.mesh, f.cells, 200e9);
    auto nu = constant_property(f.mesh, f.cells, 0.3);
    auto alpha = constant_property(f.mesh, f.cells, 1e-5);
    auto T = constant_property(f.mesh, f.cells, 393.15); // DT = 100
    sm.set_elastic(E, nu);
    sm.set_thermal_expansion(T->function(), alpha, 293.15);
    sm.add_fixed_bc(1); // x- face clamped
    sm.add_fixed_bc(2); // x+ face clamped
    sm.refresh(0.0);

    la::MatrixCSR<double> A(sm.pattern());
    la::Vector<double> b(sm.solution()->function_space()->dofmap()->index_map,
        sm.solution()->function_space()->dofmap()->index_map_bs());
    sm.assemble_steady(A, b, 0.0);

    // The AMG-preconditioned CG solve of a matrix, from a zero guess.
    auto solve = [](const la::MatrixCSR<double>& M, const la::Vector<double>& rhs,
                     la::Vector<double>& x, const char* preconditioner) {
        x.set(0);
        la::KrylovSolver<double> solver;
        solver.set_operator(M);
        solver.set_solver_type("cg");
        solver.set_preconditioner_type(preconditioner);
        solver.set_tolerances(1e-10, 1e-14, 4000);
        return solver.solve(x, rhs);
    };

    la::Vector<double> x_blocked(b.index_map(), b.bs());
    const int blocked = solve(A, b, x_blocked, "amg");

    // The same system as a scalar matrix, i.e. with every component of a
    // node a dof of its own, on a scalar index map of the physical dofs.
    la::MatrixCSR<double> S = A.to_scalar();
    const std::int32_t ndofs = 3 * b.index_map()->size_local();
    auto scalar_map = std::make_shared<hellofem::common::IndexMap>(0, ndofs);
    la::Vector<double> b_scalar(scalar_map, 1);
    for (std::size_t i = 0; i < b.array().size(); ++i)
        b_scalar.array()[i] = b.array()[i];
    la::Vector<double> x_scalar(scalar_map, 1);
    const int scalar = solve(S, b_scalar, x_scalar, "amg");

    // And the same system with no preconditioner at all. A preconditioner
    // that needs more iterations than none is not a preconditioner: an AMG
    // hierarchy whose smoother amplifies rather than smooths does exactly
    // that (amgcl's default damped Jacobi sits above the stability limit of
    // this operator and took 646 iterations here against 149 unpreconditioned
    // and 84 blocked).
    la::Vector<double> x_plain(b.index_map(), b.bs());
    const int plain = solve(A, b, x_plain, "none");

    // Both representations hold the same operator, so the two solves must
    // agree on the displacement: the difference is what the two Krylov
    // residuals buy, which is orders below what a wrong representation (a
    // mis-assembled block, a mis-expanded matrix) would show.
    double displacement = 0.0;
    double difference = 0.0;
    for (std::size_t i = 0; i < x_scalar.array().size(); ++i) {
        displacement = std::max(displacement, std::abs(x_scalar.array()[i]));
        difference = std::max(difference,
            std::abs(x_blocked.array()[i] - x_scalar.array()[i]));
    }
    INFO("blocked AMG: " << blocked << " iterations, scalar expansion: " << scalar
                         << ", unpreconditioned: " << plain
                         << ", max |u| = " << displacement
                         << ", blocked vs scalar difference = " << difference);
    REQUIRE(displacement > 0.0);
    REQUIRE(blocked > 0);
    REQUIRE(scalar > 0);
    REQUIRE(plain > 0);

    // The preconditioned solve has to beat the unpreconditioned one: that is
    // the property a divergent smoother breaks first, whatever the hierarchy.
    REQUIRE(blocked < plain);

    // The blocked hierarchy has to beat the scalar expansion clearly, but not
    // by a fixed factor: the AMG coarsening still depends on the order its
    // parallel reductions complete in, so the bound keeps room above what the
    // smoother happens to give. Measured 84 against 202 iterations for this
    // case (stable over 30 runs), i.e. a ratio of 0.42 against the bound's
    // 0.67. A reverted block-aware coarsening is what makes the two counts
    // meet, and that this still catches.
    REQUIRE(3 * blocked <= 2 * scalar);

    // The agreement of the two solves is the Krylov residual's own, amplified
    // by the inverse of the elasticity operator: it measures ~1000x the
    // relative tolerance they stop at (median 5e-13 of the displacement over
    // 30 runs here), while a wrong representation stands out by orders. The
    // bound leaves room above that measured floor.
    constexpr double agreement = 1e-6;
    REQUIRE(difference < agreement * displacement);
}

namespace {

    /// A pure-Neumann 1D Laplacian: every row sums to zero, so the constant
    /// vector is in its null space.
    la::MatrixCSR<double> make_neumann_laplacian(std::int32_t n)
    {
        auto imap = std::make_shared<hellofem::common::IndexMap>(0, n);
        la::SparsityPattern pattern(imap);
        for (std::int32_t i = 0; i < n; ++i) {
            pattern.insert(i, i);
            if (i > 0)
                pattern.insert(i, i - 1);
            if (i + 1 < n)
                pattern.insert(i, i + 1);
        }
        pattern.finalize();

        la::MatrixCSR<double> A(pattern);
        for (std::int32_t i = 0; i < n; ++i) {
            // Every row sums to zero: the constant vector is the null space.
            const double diag = (i > 0 ? 1.0 : 0.0) + (i + 1 < n ? 1.0 : 0.0);
            std::vector<std::int32_t> row {i};
            std::vector<std::int32_t> cols {i};
            std::vector<double> vals {diag};
            if (i > 0) {
                cols.push_back(i - 1);
                vals.push_back(-1.0);
            }
            if (i + 1 < n) {
                cols.push_back(i + 1);
                vals.push_back(-1.0);
            }
            A.set(vals, row, cols);
        }
        return A;
    }

} // namespace

TEST_CASE("solve_system: an iterate left at the iteration cap is refused",
    "[app][solver]")
{
    // A pure-Neumann Laplacian is singular, and a right-hand side whose mean
    // lies in its null space is one no iterate can reproduce: the Krylov
    // residual never reaches the tolerance and the solve stops at its cap,
    // leaving an intermediate iterate in `x`.
    constexpr std::int32_t n = 400;
    auto A = make_neumann_laplacian(n);
    auto imap = std::make_shared<hellofem::common::IndexMap>(0, n);
    la::Vector<double> b(imap, 1);
    b.set(1.0);
    la::Vector<double> x(imap, 1);
    x.set(0.0);

    // What the library reports for such a solve: the cap, not an exception.
    la::KrylovSolver<double> solver;
    solver.set_operator(A);
    solver.set_solver_type("cg");
    solver.set_tolerances(1e-12, 1e-14, 2000);
    REQUIRE(solver.solve(x, b) == 2000);

    // The app's own solve refuses that iterate instead of reporting it: the
    // inner linear solve throws when it stops at its cap, so a field is never
    // exported from a solve that did not converge. The nonlinear iteration
    // the app runs reads the same condition.
    la::SparsityPattern pattern(imap);
    for (std::int32_t i = 0; i < n; ++i) {
        pattern.insert(i, i);
        if (i > 0)
            pattern.insert(i, i - 1);
        if (i + 1 < n)
            pattern.insert(i, i + 1);
    }
    pattern.finalize();
    la::LinearSolver<double> inner;
    la::LinearSettings settings;
    settings.solver_type = "cg";
    settings.rtol = 1e-12;
    settings.atol = 1e-14;
    settings.max_iterations = 2000;
    la::Vector<double> guess(imap, 1);
    guess.set(0.0);
    REQUIRE_THROWS(solve_system(
        [&](la::MatrixCSR<double>& assembled, la::Vector<double>& rhs) {
            assembled = A;
            rhs = b;
        },
        guess, pattern, inner, settings));
}

TEST_CASE("SolidMechanics: a linear displacement is exact on a tetrahedral mesh",
    "[app][physics]")
{
    // The blocked space on tetrahedra. Every other solid test uses a box of
    // hexahedra, and the field solver's patch tests on tetrahedra are all
    // scalar, so this is the one combination neither covers. A linear
    // displacement is in every Lagrange space, so the discrete solution must
    // reproduce it exactly at any order — including at the orders where the
    // element's dofs need permuting into the reference order, which for a
    // tetrahedron first happens at the third.
    using hellofem::mesh::CellType;
    const std::vector<double> x {0, 0, 0, /**/ 1, 0, 0, /**/ 0, 1, 0,
        /**/ 1, 1, 0, /**/ 0, 0, 1, /**/ 1, 0, 1, /**/ 0, 1, 1, /**/ 1, 1, 1};
    const std::vector<std::int64_t> cells {0, 1, 3, 7, /**/ 0, 1, 5, 7,
        /**/ 0, 2, 3, 7, /**/ 0, 2, 6, 7, /**/ 0, 4, 5, 7, /**/ 0, 4, 6, 7};
    auto mesh = std::make_shared<hellofem::mesh::Mesh<double>>(
        hellofem::mesh::create_mesh(std::span<const std::int64_t>(cells),
            CellType::tetrahedron, x, 3));
    auto boundary = hellofem::app::test::boundary_tags(*mesh);
    const std::size_t nc = mesh->topology()->index_map(3)->size_local();
    std::vector<std::int32_t> cell_idx(nc);
    std::iota(cell_idx.begin(), cell_idx.end(), 0);
    auto cell_tags = std::make_shared<hellofem::mesh::MeshTags<int>>(
        mesh->topology(), 3, std::move(cell_idx), std::vector<int>(nc, 1), "cells");

    for (const int order : {1, 2, 3, 4}) {
        SolidMechanicsSolver sm(mesh, boundary, cell_tags, order);
        sm.set_elastic(constant_property(mesh, cell_tags, 200e9),
            constant_property(mesh, cell_tags, 0.3));
        sm.add_fixed_bc(1); // x- face clamped
        sm.refresh(0.0);

        auto u = sm.solution();
        const std::int32_t nnodes = sm.solution()->function_space()->dofmap()->index_map->size_local();
        auto coords = sm.solution()->function_space()->tabulate_dof_coordinates(false);

        // u = (x, 0, 0): linear, and zero on the clamped face.
        la::Vector<double> exact(u->x()->index_map(), u->x()->bs());
        for (std::int32_t d = 0; d < nnodes; ++d) {
            exact.array()[static_cast<std::size_t>(3 * d)]
                = coords[static_cast<std::size_t>(9 * d)];
            exact.array()[static_cast<std::size_t>(3 * d + 1)] = 0.0;
            exact.array()[static_cast<std::size_t>(3 * d + 2)] = 0.0;
        }

        la::MatrixCSR<double> A(sm.pattern());
        la::Vector<double> b(u->x()->index_map(), u->x()->bs());
        sm.assemble_steady(A, b, 0.0);
        A.mult(exact, b); // solve for the field we already know

        la::Vector<double> sol(u->x()->index_map(), u->x()->bs());
        la::KrylovSolver<double> solver;
        solver.set_operator(A);
        solver.set_solver_type("cg");
        solver.set_preconditioner_type("amg");
        solver.set_tolerances(1e-10, 0.0, 5000);
        const int iters = solver.solve(sol, b);

        double err = 0.0, mag = 0.0;
        for (std::size_t i = 0; i < sol.array().size(); ++i) {
            err = std::max(err, std::abs(sol.array()[i] - exact.array()[i]));
            mag = std::max(mag, std::abs(exact.array()[i]));
        }
        INFO("order " << order << ": " << iters << " iterations, max error "
                      << err << " against max |u| = " << mag);
        REQUIRE(err < 1e-8 * mag);
    }
}

namespace {

    /// Krylov iterations of the electrostatic system on an `n`-divided unit
    /// box at the given element order.
    int electrostatics_iterations(int n, int order)
    {
        auto f = make_box_fixture({0, 0, 0}, {1, 1, 1}, {n, n, n});
        ElectrostaticsSolver es(f.mesh, f.boundary, f.cells, order);
        es.set_conductivity(constant_property(f.mesh, f.cells, 1.0));
        es.add_voltage_bc(2, ScalarExpression(1.0));
        es.add_voltage_bc(1, ScalarExpression(0.0));
        es.refresh(0.0);

        auto x = es.solution()->x();
        la::MatrixCSR<double> A(es.pattern());
        la::Vector<double> b(x->index_map(), x->bs());
        es.assemble_steady(A, b, 0.0);

        la::KrylovSolver<double> solver;
        solver.set_operator(A);
        solver.set_solver_type("cg");
        solver.set_preconditioner_type("amg");
        solver.set_tolerances(1e-8, 0.0, 5000);
        return solver.solve(*x, b);
    }

} // namespace

TEST_CASE("AMG: the iteration count does not grow with the problem",
    "[app][solver]")
{
    // What a multigrid preconditioner is for: its Krylov iteration count is
    // set by the operator and the hierarchy it builds, not by how many
    // unknowns the mesh carries. Smoothed aggregation with amgcl's defaults
    // is not — it groups whole neighbourhoods into a single aggregate and
    // interpolates from one constant per aggregate — and the count climbs
    // with the problem: 22 iterations at 4913 unknowns against 41 at 35937
    // and 51 at 68921. amgcl's own solver reproduces those counts exactly on
    // the same matrix, so it is the coarsening the count measures and not
    // this wrapper. Classical coarsening holds 7 and 6 over the same pair.
    const int coarse = electrostatics_iterations(8, 2);
    const int fine = electrostatics_iterations(16, 2);
    INFO("AMG iterations: 4913 unknowns -> " << coarse << ", 35937 -> " << fine);
    REQUIRE(fine <= coarse + 4);
}
