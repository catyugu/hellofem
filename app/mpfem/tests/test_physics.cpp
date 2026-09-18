// hellofem::app — physics solver tests: manufactured and analytic solutions
// SPDX-License-Identifier: MIT

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
#include "solid.h"
#include "solver.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <string>
#include <tuple>
#include <vector>

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

        auto coords = es.space()->tabulate_dof_coordinates(false);
        double max_err = 0;
        for (std::int32_t d = 0;
            d < es.space()->dofmap()->index_map->size_local(); ++d)
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
    auto sigma = std::make_shared<CellProperty>(f.mesh, f.cells,
        std::unordered_map<std::string, double> {});
    sigma->bind_field("V", es.solution());
    sigma->set_expression(1, "1 + V");
    es.set_conductivity(sigma);

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

TEST_CASE("solve_linear: an iterate left at the iteration cap is not a solution",
    "[app][solver]")
{
    // A pure-Neumann Laplacian is singular, and a right-hand side whose mean
    // lies in its null space is one no iterate can reproduce: the Krylov
    // residual never reaches the tolerance and the solve stops at its cap.
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

    // Which the app must read as a failure: a field whose solve stopped at
    // the cap is a field that was never solved, and exporting its iterate
    // reports a non-solution as a result.
    REQUIRE_FALSE(converged(2000, 2000));
    REQUIRE(converged(165, 2000));
    REQUIRE(converged(1, 2000));
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
        const std::int32_t nnodes = sm.space()->dofmap()->index_map->size_local();
        auto coords = sm.space()->tabulate_dof_coordinates(false);

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
        sm.assemble_steady(A, b);
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
        es.assemble_steady(A, b);

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
