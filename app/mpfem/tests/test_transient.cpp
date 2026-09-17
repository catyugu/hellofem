// hellofem::app — time stepping and nonlinear iteration tests
// SPDX-License-Identifier: MIT

#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "fixture.h"
#include "time_scheme.h"
#include "transient.h"

#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>

namespace la = hellofem::la;

using Catch::Approx;
using namespace hellofem::app;

namespace {

    std::unordered_map<std::string, double> no_params()
    {
        return {};
    }

    /// Heat equation with k = rho cp = 1 and the manufactured solution
    /// T(x,t) = t^3 (uniform in space, so the source is 3 t^2 and the finite
    /// element solution is exact in space): the error measured here is the
    /// time scheme's alone. Both x faces carry the Dirichlet data t^3.
    double manufactured_error(std::string_view scheme, double dt, double t_end)
    {
        auto box = test::make_box_fixture({0, 0, 0}, {1, 0.2, 0.2}, {4, 1, 1});
        auto solver = std::make_shared<HeatTransferSolver>(
            box.mesh, box.boundary, box.cells, 1);
        solver->set_conductivity(test::constant_property(box.mesh, box.cells, 1.0));
        solver->set_thermal_mass(test::constant_property(box.mesh, box.cells, 1.0));
        auto source = std::make_shared<CellProperty>(box.mesh, box.cells,
            no_params());
        source->set_expression(1, "3*t*t");
        solver->set_source(source);
        solver->add_temperature_bc(1, ScalarExpression("t*t*t"));
        solver->add_temperature_bc(2, ScalarExpression("t*t*t"));
        solver->set_initial_temperature(ScalarExpression(0.0));
        solver->apply_initial_condition();

        HeatTimeStepper stepper(solver, make_time_scheme(scheme));
        stepper.start(0.0);
        const int steps = static_cast<int>(std::llround(t_end / dt));
        for (int n = 1; n <= steps; ++n)
            stepper.step(static_cast<double>(n) * dt);

        double error = 0.0;
        const double exact = t_end * t_end * t_end;
        for (double value : solver->solution()->x()->array())
            error = std::max(error, std::abs(value - exact));
        return error;
    }

    /// Observed order of accuracy between two successive refinements.
    double observed_order(double coarse, double fine)
    {
        return std::log2(coarse / fine);
    }

    /// Steady nonlinear rod: -d/dx(k(T) dT/dx) = 0 with a linear k(T) law.
    struct NonlinearRod {
        test::BoxFixture box;
        std::shared_ptr<HeatTransferSolver> solver;

        explicit NonlinearRod(double alpha = -0.001)
        {
            box = test::make_box_fixture({0, 0, 0}, {1, 0.2, 0.2}, {8, 1, 1});
            solver = std::make_shared<HeatTransferSolver>(
                box.mesh, box.boundary, box.cells, 1);
            auto k = std::make_shared<CellProperty>(box.mesh, box.cells,
                std::unordered_map<std::string, double> {
                    {"k0", 45.0}, {"alpha_k", alpha}, {"Tref", 293.15}});
            k->bind_field("T", solver->solution());
            k->set_expression(1, "k0*(1+alpha_k*(T-Tref))");
            solver->set_conductivity(k);
            solver->add_temperature_bc(1, ScalarExpression(293.15));
            solver->add_temperature_bc(2, ScalarExpression(373.15));
        }

        int solve()
        {
            return solve_system(
                [&](la::MatrixCSR<double>& A, la::Vector<double>& b) {
                    solver->refresh(0.0);
                    solver->assemble_steady(A, b);
                },
                *solver->solution()->x(), solver->pattern(), solver->nonlinear());
        }
    };

} // namespace

TEST_CASE("TimeScheme: weights, consistency and start-up", "[app][transient]")
{
    const double dt = 0.5;
    TimeWeights w;

    auto bdf1 = make_time_scheme("bdf1");
    REQUIRE(bdf1->order() == 1);
    bdf1->weights(dt, 1, w);
    REQUIRE(w.a.size() == 2);
    REQUIRE(w.a[0] == Approx(2.0));
    REQUIRE(w.a[1] == Approx(-2.0));
    REQUIRE(w.b[0] == Approx(1.0));
    REQUIRE(w.b[1] == Approx(0.0));
    REQUIRE(w.c_new == Approx(1.0));

    auto cn = make_time_scheme("cn");
    REQUIRE(cn->order() == 2);
    cn->weights(dt, 1, w);
    REQUIRE(w.b[0] == Approx(0.5));
    REQUIRE(w.b[1] == Approx(0.5));
    REQUIRE(w.c_new == Approx(0.5));
    REQUIRE(w.c_old == Approx(0.5));

    auto bdf2 = make_time_scheme("bdf2");
    REQUIRE(bdf2->order() == 2);
    bdf2->weights(dt, 2, w);
    REQUIRE(w.a.size() == 3);
    REQUIRE(w.a[0] == Approx(3.0)); // 3 / (2 dt)
    REQUIRE(w.a[1] == Approx(-4.0)); // -2 / dt
    REQUIRE(w.a[2] == Approx(1.0)); // 1 / (2 dt)
    REQUIRE(w.b[0] == Approx(1.0));
    REQUIRE(w.b[1] == Approx(0.0));

    // A scheme that lacks its history falls back to a first-order step.
    bdf2->weights(dt, 1, w);
    REQUIRE(w.a.size() == 2);
    REQUIRE(w.a[0] == Approx(2.0));

    // Consistency: the mass weights sum to zero (a constant state has no
    // time derivative) and the stiffness weights to one.
    for (const std::string& name : time_scheme_names()) {
        auto scheme = make_time_scheme(name);
        scheme->weights(dt, scheme->levels(), w);
        double sum_a = 0, sum_b = 0;
        for (double value : w.a)
            sum_a += value;
        for (double value : w.b)
            sum_b += value;
        INFO("scheme " << name << ": sum(a) = " << sum_a << ", sum(b) = " << sum_b);
        REQUIRE(sum_a == Approx(0.0).margin(1e-12));
        REQUIRE(sum_b == Approx(1.0).margin(1e-12));
        REQUIRE(w.c_new + w.c_old == Approx(1.0).margin(1e-12));
    }

    REQUIRE_THROWS(make_time_scheme("rk4"));
}

TEST_CASE("Transient heat: time scheme order on a manufactured solution", "[app][transient]")
{
    // The time discretization error is measured alone (the manufactured
    // solution is spatially uniform). Backward Euler is first order, while
    // Crank-Nicolson and BDF2 are second order; the BDF2 start-up step is
    // first order but its local error is one order higher, so the global
    // order stays two.
    const double t_end = 1.0;
    const std::vector<double> steps {0.05, 0.025, 0.0125};
    for (const auto& [scheme, expected] :
        std::vector<std::pair<std::string, double>> {
            {"bdf1", 1.0}, {"cn", 2.0}, {"bdf2", 2.0}}) {
        std::vector<double> errors;
        for (double dt : steps)
            errors.push_back(manufactured_error(scheme, dt, t_end));
        const double p1 = observed_order(errors[0], errors[1]);
        const double p2 = observed_order(errors[1], errors[2]);
        INFO("scheme " << scheme << ": errors = " << errors[0] << ", "
                       << errors[1] << ", " << errors[2] << " -> orders "
                       << p1 << ", " << p2 << " (expected " << expected << ")");
        REQUIRE(errors[0] > 0.0);
        REQUIRE(p1 > expected - 0.25);
        REQUIRE(p2 > expected - 0.25);
    }
}

TEST_CASE("Transient heat: a solution-independent nonlinear law matches the linear one",
    "[app][transient]")
{
    // k(T) = k0 with alpha_k = 0 still reads the field, so the nonlinear
    // path (Anderson-accelerated Picard) runs; it must reproduce the
    // constant-property solve.
    auto box = test::make_box_fixture({0, 0, 0}, {1, 0.2, 0.2}, {4, 1, 1});
    auto run = [&](bool field_dependent) {
        auto solver = std::make_shared<HeatTransferSolver>(
            box.mesh, box.boundary, box.cells, 1);
        auto k = std::make_shared<CellProperty>(box.mesh, box.cells,
            std::unordered_map<std::string, double> {
                {"k0", 1.0}, {"alpha_k", 0.0}, {"Tref", 0.0}});
        if (field_dependent) {
            k->bind_field("T", solver->solution());
            k->set_expression(1, "k0*(1+alpha_k*(T-Tref))");
        }
        else
            k->set_expression(1, "k0");
        solver->set_conductivity(k);
        solver->set_thermal_mass(test::constant_property(box.mesh, box.cells, 1.0));
        auto source = std::make_shared<CellProperty>(box.mesh, box.cells,
            no_params());
        source->set_expression(1, "3*t*t");
        solver->set_source(source);
        solver->add_temperature_bc(1, ScalarExpression("t*t*t"));
        solver->add_temperature_bc(2, ScalarExpression("t*t*t"));
        solver->set_initial_temperature(ScalarExpression(0.0));
        solver->apply_initial_condition();
        REQUIRE(solver->nonlinear() == field_dependent);

        HeatTimeStepper stepper(solver, make_time_scheme("bdf2"));
        stepper.start(0.0);
        for (int n = 1; n <= 20; ++n)
            stepper.step(0.05 * n);
        return solver;
    };

    auto linear = run(false);
    auto nonlinear = run(true);
    double max_diff = 0;
    for (std::size_t i = 0; i < linear->solution()->x()->array().size(); ++i)
        max_diff = std::max(max_diff,
            std::abs(linear->solution()->x()->array()[i]
                - nonlinear->solution()->x()->array()[i]));
    INFO("linear vs nonlinear max difference = " << max_diff);
    REQUIRE(max_diff < 1e-6);
}

TEST_CASE("Nonlinear iteration: a warm start from the previous solution converges at once",
    "[app][transient]")
{
    // The previous solution is the initial guess of the next solve (the
    // transient stepper hands it in as the state); a converged state needs
    // no iteration at all, while a cold start needs several.
    NonlinearRod rod(-0.005);
    const int cold = rod.solve();
    const la::Vector<double> converged = *rod.solver->solution()->x();
    const int warm = rod.solve();
    double drift = 0;
    for (std::size_t i = 0; i < converged.array().size(); ++i)
        drift = std::max(drift,
            std::abs(converged.array()[i]
                - rod.solver->solution()->x()->array()[i]));
    INFO("cold = " << cold << " iterations, warm = " << warm
                   << " iterations, drift = " << drift);
    REQUIRE(cold >= 3);
    REQUIRE(warm <= 2);
    // The solve leaves an already converged state alone to within the
    // nonlinear tolerance (a residual test, hence the loose bound).
    REQUIRE(drift < 1e-5);

    // The solution is the analytic profile of the nonlinear rod.
    const double alpha = -0.005, t_ref = 293.15;
    const double t0 = 293.15, t1 = 373.15;
    auto u = [&](double T) {
        return (T - t_ref) + 0.5 * alpha * (T - t_ref) * (T - t_ref);
    };
    auto exact = [&](double x) {
        const double target = u(t0) + (u(t1) - u(t0)) * x;
        return t_ref + (-1.0 + std::sqrt(1.0 + 2.0 * alpha * target)) / alpha;
    };
    const auto coords = rod.solver->space()->tabulate_dof_coordinates(false);
    double max_err = 0;
    for (std::int32_t d = 0;
        d < rod.solver->space()->dofmap()->index_map->size_local(); ++d)
        max_err = std::max(max_err,
            std::abs(rod.solver->solution()->x()->array()[static_cast<std::size_t>(d)]
                - exact(coords[3 * d])));
    INFO("analytic nonlinear rod max error = " << max_err);
    REQUIRE(max_err < 1e-2);
}
