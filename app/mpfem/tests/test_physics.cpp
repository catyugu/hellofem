// hellofem::app — physics solver tests: manufactured and analytic solutions
// SPDX-License-Identifier: MIT

#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "electric.h"
#include "fixture.h"
#include "heat.h"
#include "la/KrylovSolver.h"
#include "mesh/generation.h"
#include "mesh/utils.h"
#include "solid.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <numeric>

namespace la = hellofem::la;

using Catch::Approx;
using namespace hellofem::app;
using hellofem::app::test::constant_property;
using hellofem::app::test::make_box_fixture;

TEST_CASE("CellProperty: sparse domains and field-dependent values", "[app][physics]")
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
    const auto coords = ht.space()->tabulate_dof_coordinates(false);
    for (std::int32_t d = 0; d < ht.space()->dofmap()->index_map->size_local(); ++d)
        field.x()->array()[static_cast<std::size_t>(d)] = coords[3 * d];

    CellProperty property(fixture.mesh, tags, {});
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

TEST_CASE("CellProperty: a law reading no field does not evaluate a bound one",
    "[app][physics]")
{
    // A physics publishes its solution for every material law that may read
    // it. A law that reads none must not pay for evaluating it — and must
    // not be refused a field whose value shape it never asked for.
    auto f = make_box_fixture({0, 0, 0}, {1, 1, 1}, {2, 2, 1});
    SolidMechanicsSolver sm(f.mesh, f.boundary, f.cells, 1); // vector solution
    CellProperty property(f.mesh, f.cells, {});
    property.bind_field("u", sm.solution());
    property.set_expression(1, "3.0");
    REQUIRE_FALSE(property.field_dependent());

    property.update(0.0);

    const auto& dofmap = *property.function()->function_space()->dofmap();
    REQUIRE(property.function()->x()->array()[static_cast<std::size_t>(
        dofmap.cell_dofs(0).front())]
        == Approx(3.0));
}

TEST_CASE("Electrostatics: -div(sigma grad V)=0 with V=V0 on x+, V=0 on x-", "[app][physics]")
{
    // 1x1x1 box, single layer in z. V solves Laplace; with V=1 on x+, 0 on x-,
    // the solution is V = x (linear), sigma = 1.
    auto f = make_box_fixture({0, 0, 0}, {1, 1, 1}, {4, 4, 1});
    ElectrostaticsSolver es(f.mesh, f.boundary, f.cells, 1);
    auto sigma = constant_property(f.mesh, f.cells, 1.0);
    es.set_conductivity(sigma);
    es.add_voltage_bc(2, ScalarExpression(1.0)); // x+ : V=1
    es.add_voltage_bc(1, ScalarExpression(0.0)); // x- : V=0

    es.solve_steady(0.0);
    auto V = es.solution();

    // V = x at every dof coordinate.
    auto coords = es.space()->tabulate_dof_coordinates(false);
    double max_err = 0;
    double v_max = -1e9, v_min = 1e9;
    for (std::int32_t d = 0; d < es.space()->dofmap()->index_map->size_local(); ++d) {
        const double x = coords[3 * d];
        const double val = V->x()->array()[static_cast<std::size_t>(d)];
        max_err = std::max(max_err, std::abs(val - x));
        v_max = std::max(v_max, val);
        v_min = std::min(v_min, val);
    }
    INFO("max error = " << max_err << ", V range [" << v_min << "," << v_max << "]");
    REQUIRE(max_err < 1e-10);
    REQUIRE(v_max == Catch::Approx(1.0).margin(1e-9));
    REQUIRE(v_min == Catch::Approx(0.0).margin(1e-9));
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
    auto sigma = std::make_shared<CellProperty>(f.mesh, f.cells,
        std::unordered_map<std::string, double> {});
    sigma->bind_field("V", es.solution());
    sigma->set_expression(1, "1 + V");
    es.set_conductivity(sigma);
    REQUIRE(es.nonlinear());

    es.solve_steady(0.0);

    auto coords = es.space()->tabulate_dof_coordinates(false);
    double err_exact = 0, err_linear = 0;
    for (std::int32_t d = 0; d < es.space()->dofmap()->index_map->size_local(); ++d) {
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
    auto h = constant_property(f.mesh, f.cells, 1.0);
    auto t_inf = constant_property(f.mesh, f.cells, 0.0);
    ht.set_conductivity(k);
    ht.add_temperature_bc(1, ScalarExpression(1.0)); // x- : T=1
    ht.add_convection(2, h, t_inf); // x+ : h=1, Tinf=0

    ht.solve_steady(0.0);
    auto T = ht.solution();

    auto coords = ht.space()->tabulate_dof_coordinates(false);
    double max_err = 0;
    for (std::int32_t d = 0; d < ht.space()->dofmap()->index_map->size_local(); ++d) {
        const double x = coords[3 * d];
        const double exact = 1.0 - 0.5 * x;
        max_err = std::max(max_err,
            std::abs(T->x()->array()[static_cast<std::size_t>(d)] - exact));
    }
    INFO("heat max error = " << max_err);
    REQUIRE(max_err < 1e-8);
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
    auto k = std::make_shared<CellProperty>(f.mesh, f.cells,
        std::unordered_map<std::string, double> {
            {"k0", k0}, {"alpha_k", a}, {"Tref", t_ref}});
    k->bind_field("T", ht.solution());
    k->set_expression(1, "k0*(1+alpha_k*(T-Tref))");
    ht.set_conductivity(k);
    ht.add_temperature_bc(1, ScalarExpression(t0));
    ht.add_temperature_bc(2, ScalarExpression(t1));
    REQUIRE(ht.nonlinear());

    ht.solve_steady(0.0);

    auto coords = ht.space()->tabulate_dof_coordinates(false);
    double max_err = 0;
    for (std::int32_t d = 0; d < ht.space()->dofmap()->index_map->size_local(); ++d) {
        const double x = coords[3 * d];
        max_err = std::max(max_err,
            std::abs(ht.solution()->x()->array()[static_cast<std::size_t>(d)]
                - exact(x)));
    }
    INFO("nonlinear k(T) max error = " << max_err);
    REQUIRE(max_err < 1e-3);
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
    auto coords = sm.space()->tabulate_dof_coordinates(false);
    double ux_max = 0, ux_at_1 = -1e9;
    double max_lat = 0;
    for (std::int32_t d = 0; d < sm.space()->dofmap()->index_map->size_local(); ++d) {
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
    la::Vector<double> b(sm.space()->dofmap()->index_map,
        sm.space()->dofmap()->index_map_bs());
    sm.assemble_steady(A, b);

    // The AMG-preconditioned CG solve of a matrix, from a zero guess.
    auto solve = [](const la::MatrixCSR<double>& M, const la::Vector<double>& rhs,
                     la::Vector<double>& x) {
        x.set(0);
        la::KrylovSolver<double> solver;
        solver.set_operator(M);
        solver.set_solver_type("cg");
        solver.set_preconditioner_type("amg");
        solver.set_tolerances(1e-10, 1e-14, 4000);
        return solver.solve(x, rhs);
    };

    la::Vector<double> x_blocked(b.index_map(), b.bs());
    const int blocked = solve(A, b, x_blocked);

    // The same system as a scalar matrix, i.e. with every component of a
    // node a dof of its own, on a scalar index map of the physical dofs.
    la::MatrixCSR<double> S = A.to_scalar();
    const std::int32_t ndofs = 3 * b.index_map()->size_local();
    auto scalar_map = std::make_shared<hellofem::common::IndexMap>(0, ndofs);
    la::Vector<double> b_scalar(scalar_map, 1);
    for (std::size_t i = 0; i < b.array().size(); ++i)
        b_scalar.array()[i] = b.array()[i];
    la::Vector<double> x_scalar(scalar_map, 1);
    const int scalar = solve(S, b_scalar, x_scalar);

    INFO("blocked AMG: " << blocked << " iterations, scalar expansion: "
                         << scalar);
    REQUIRE(blocked > 0);
    REQUIRE(scalar > 0);
    REQUIRE(2 * blocked <= scalar);
    for (std::size_t i = 0; i < x_scalar.array().size(); ++i)
        REQUIRE(std::abs(x_blocked.array()[i] - x_scalar.array()[i]) < 1e-12);
}
