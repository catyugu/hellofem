// hellofem::app — time stepping and nonlinear iteration tests
// SPDX-License-Identifier: MIT

#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "fixture.h"
#include "heat.h"
#include "time_scheme.h"
#include "transient.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace la = hellofem::la;

using Catch::Approx;
using namespace hellofem::app;

namespace {

    std::unordered_map<std::string, double> no_params()
    {
        return {};
    }

    /// A manufactured time-dependent heat problem: the heat equation with
    /// k = rho cp = 1 and a solution that is uniform in space, `T(x,t) =
    /// exact(t)`, driven by its time derivative as the source and held on
    /// both x faces. The spatial discretization is exact, so the error the
    /// levels carry is the time scheme's alone.
    struct Manufactured {
        std::string exact; // the solution, as a model expression in t
        std::string derivative; // its time derivative, the source
        std::function<double(double)> value; // the same solution in C++
    };

    /// The cubic solution T = t^3, whose source is 3 t^2.
    Manufactured cubic_problem()
    {
        return Manufactured {"t*t*t", "3*t*t", [](double t) { return t * t * t; }};
    }

    /// The solver of `problem`, at its initial state of t = 0.
    std::shared_ptr<HeatTransferSolver> manufactured_solver(const Manufactured& problem)
    {
        auto box = test::make_box_fixture({0, 0, 0}, {1, 0.2, 0.2}, {4, 1, 1});
        auto solver = std::make_shared<HeatTransferSolver>(
            box.mesh, box.boundary, box.cells, 1);
        solver->set_conductivity(test::constant_property(box.mesh, box.cells, 1.0));
        solver->set_thermal_mass(test::constant_property(box.mesh, box.cells, 1.0));
        auto source = std::make_shared<CellProperty>(box.mesh, box.cells,
            no_params());
        source->set_expression(1, problem.derivative);
        solver->set_source(source);
        solver->add_temperature_bc(1, ScalarExpression(problem.exact));
        solver->add_temperature_bc(2, ScalarExpression(problem.exact));
        solver->set_initial_temperature(ScalarExpression(0.0));
        solver->apply_initial_condition();
        return solver;
    }

    /// The largest error of a solved state against the exact solution at `t`.
    double state_error(const Manufactured& problem,
        const std::shared_ptr<HeatTransferSolver>& solver, double t)
    {
        double error = 0.0;
        const double exact = problem.value(t);
        for (double value : solver->solution()->x()->array())
            error = std::max(error, std::abs(value - exact));
        return error;
    }

    /// Heat equation with k = rho cp = 1 and the manufactured solution
    /// T(x,t) = t^3 (uniform in space, so the source is 3 t^2 and the finite
    /// element solution is exact in space): the error measured here is the
    /// time scheme's alone. Both x faces carry the Dirichlet data t^3.
    double manufactured_error(
        std::string_view scheme, int order, double dt, double t_end)
    {
        const Manufactured problem = cubic_problem();
        auto solver = manufactured_solver(problem);

        TimeStepper stepper(*solver, TimeSettings {std::string(scheme), false, 1e-3});
        stepper.start(0.0);
        const int steps = static_cast<int>(std::llround(t_end / dt));
        double worst = 0.0;
        for (int n = 1; n <= steps; ++n) {
            const double t = static_cast<double>(n) * dt;
            stepper.step(t, order);
            worst = std::max(worst, state_error(problem, solver, t));
        }
        return worst;
    }

    /// A run of an adaptive or a fixed stepping: the worst error over its
    /// levels — each level against the exact solution at its own time, so a
    /// problem that settles is not judged by its end state alone — the steps
    /// taken, and the steps the error test threw away.
    struct RunResult {
        double worst_error = 0.0;
        int steps = 0;
        int rejected = 0;
    };

    /// Integrate `problem` to `t_end` with a fixed second-order step `dt`.
    RunResult run_uniform(const Manufactured& problem, double dt, double t_end)
    {
        auto solver = manufactured_solver(problem);
        TimeStepper stepper(*solver, TimeSettings {"bdf2", false, 1e-3});
        stepper.start(0.0);

        RunResult run;
        double t = 0.0;
        while (t < t_end) {
            const double step = std::min(dt, t_end - t);
            t = step < t_end - t ? t + step : t_end;
            stepper.step(t, 2);
            run.worst_error = std::max(run.worst_error, state_error(problem, solver, t));
            ++run.steps;
        }
        return run;
    }

    /// Integrate `problem` to `t_end` with the adaptive controller of
    /// `CaseScheduler::advance_adaptive`: its rules are the ones under test,
    /// driven here over the one field.
    RunResult run_adaptive(
        const Manufactured& problem, double tolerance, double t_end)
    {
        auto solver = manufactured_solver(problem);
        TimeStepper stepper(*solver, TimeSettings {"bdf2", true, tolerance});
        stepper.start(0.0);

        const int max_order = find_time_scheme("bdf2").order;
        // The scheduler starts at a thousandth of the span (a first step has
        // no history for the estimate), and doubles from there.
        double dt = t_end / 1000.0;
        double t = 0.0;
        int order = 1;
        RunResult run;
        while (t < t_end) {
            const double remaining = t_end - t;
            const double h = std::min(dt, remaining);
            stepper.step(h < remaining ? t + h : t_end, order);
            const double error = stepper.error();
            if (error > 1.0) {
                stepper.undo();
                dt = h * step_factor(error, order);
                order = std::max(1, order - 1);
                ++run.rejected;
                continue;
            }
            t = h < remaining ? t + h : t_end;
            const int next
                = next_order(order, max_order, stepper.scaled_derivatives());
            dt = h * step_factor(error, order);
            order = next;
            run.worst_error = std::max(run.worst_error, state_error(problem, solver, t));
            ++run.steps;
        }
        return run;
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

        int solve() { return solver->solve_steady(0.0); }
    };

} // namespace

TEST_CASE("TimeScheme: the BDF weights follow the steps, the controller the error",
    "[app][transient]")
{
    const double dt = 0.5;

    TimeWeights w = bdf_weights(1, TimeSteps {dt, dt});
    REQUIRE(w.a.size() == 2);
    REQUIRE(w.a[0] == Approx(2.0));
    REQUIRE(w.a[1] == Approx(-2.0));
    REQUIRE(w.b[0] == Approx(1.0));
    REQUIRE(w.b[1] == Approx(0.0));
    REQUIRE(w.c_new == Approx(1.0));

    w = cn_weights(dt);
    REQUIRE(w.b[0] == Approx(0.5));
    REQUIRE(w.b[1] == Approx(0.5));
    REQUIRE(w.c_new == Approx(0.5));
    REQUIRE(w.c_old == Approx(0.5));

    w = bdf_weights(2, TimeSteps {dt, dt});
    REQUIRE(w.a.size() == 3);
    REQUIRE(w.a[0] == Approx(3.0)); // 3 / (2 dt)
    REQUIRE(w.a[1] == Approx(-4.0)); // -2 / dt
    REQUIRE(w.a[2] == Approx(1.0)); // 1 / (2 dt)
    REQUIRE(w.b[0] == Approx(1.0));
    REQUIRE(w.b[1] == Approx(0.0));

    // A second-order step with no step behind it falls back to backward
    // Euler: the quadratic it differentiates wants a level it has not got.
    w = bdf_weights(2, TimeSteps {dt, 0.0});
    REQUIRE(w.a.size() == 2);
    REQUIRE(w.a[0] == Approx(2.0));
    REQUIRE(w.a[1] == Approx(-2.0));

    // Consistency at any step ratio: the mass weights sum to zero (a constant
    // state has no time derivative), the stiffness weights to one, and the
    // load coefficients split the load over the two levels.
    for (const double dt_previous : {dt, dt / 2.0, dt / 3.0}) {
        for (const int order : {1, 2}) {
            w = bdf_weights(order, TimeSteps {dt, dt_previous});
            double sum_a = 0, sum_b = 0;
            for (double value : w.a)
                sum_a += value;
            for (double value : w.b)
                sum_b += value;
            INFO("bdf order " << order << " with steps " << dt << " and "
                              << dt_previous << ": sum(a) = " << sum_a
                              << ", sum(b) = " << sum_b);
            REQUIRE(sum_a == Approx(0.0).margin(1e-12));
            REQUIRE(sum_b == Approx(1.0).margin(1e-12));
        }
        w = cn_weights(dt);
        REQUIRE(w.c_new + w.c_old == Approx(1.0).margin(1e-12));
    }

    // The weights differentiate the polynomial through the levels exactly:
    // for u = 1 + 2t + 3t^2 the derivative at the new level is 2 + 6t, and
    // the second-order weights reproduce it at an uneven pair of steps too.
    // This is what keeps a run second order when the step size changes.
    const double h = 0.3, h_previous = 0.1;
    w = bdf_weights(2, TimeSteps {h, h_previous});
    const auto u = [](double t) { return 1.0 + 2.0 * t + 3.0 * t * t; };
    const double t_new = 1.0 + h, t_old = 1.0, t_older = 1.0 - h_previous;
    REQUIRE(w.a[0] * u(t_new) + w.a[1] * u(t_old) + w.a[2] * u(t_older)
        == Approx(2.0 + 6.0 * t_new).margin(1e-12));

    // The controller: a step that met the tolerance is left alone until it is
    // well inside, doubled when it is, and one that missed it is repeated
    // smaller by the asymptotic dependence of the error on the step.
    REQUIRE(step_factor(0.5, 2) == Approx(1.0));
    REQUIRE(step_factor(1.0 / 20.0, 1) == Approx(2.0));
    REQUIRE(step_factor(2.0, 1) == Approx(0.9 / std::sqrt(2.0)));
    REQUIRE(step_factor(2.0, 2) == Approx(0.9 / std::cbrt(2.0)));
    REQUIRE(step_factor(1e6, 1) == Approx(0.1));

    // The order follows the monotonicity of the scaled derivative norms:
    // the higher order is worth taking while the terms of the expansion keep
    // shrinking.
    REQUIRE(next_order(1, 2, std::vector<double> {1.0, 0.5}) == 2);
    REQUIRE(next_order(1, 2, std::vector<double> {1.0, 2.0}) == 1);
    REQUIRE(next_order(2, 2, std::vector<double> {1.0, 2.0, 4.0}) == 1);
    REQUIRE(next_order(2, 2, std::vector<double> {1.0, 0.5, 0.25}) == 2);
    REQUIRE(next_order(2, 2, std::vector<double> {}) == 2);

    REQUIRE_THROWS(find_time_scheme("rk4"));
    REQUIRE(find_time_scheme("BDF2").family == TimeFamily::bdf);
    REQUIRE(find_time_scheme("cn").family == TimeFamily::crank_nicolson);

    // The step control is not the scheme: adaptive steps are the BDF family's
    // alone, and asking another family for them is an error rather than a
    // quietly wrong estimate.
    auto solver = manufactured_solver(cubic_problem());
    REQUIRE_NOTHROW(TimeStepper(*solver, TimeSettings {"bdf2", true, 1e-3}));
    REQUIRE_NOTHROW(TimeStepper(*solver, TimeSettings {"bdf1", true, 1e-3}));
    REQUIRE_THROWS(TimeStepper(*solver, TimeSettings {"cn", true, 1e-3}));
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
    struct Use {
        std::string scheme;
        int order;
        double expected;
    };
    for (const Use& use : std::vector<Use> {
             {"bdf1", 1, 1.0}, {"cn", 2, 2.0}, {"bdf2", 2, 2.0}}) {
        std::vector<double> errors;
        for (double dt : steps)
            errors.push_back(manufactured_error(use.scheme, use.order, dt, t_end));
        const double p1 = observed_order(errors[0], errors[1]);
        const double p2 = observed_order(errors[1], errors[2]);
        INFO("scheme " << use.scheme << ": errors = " << errors[0] << ", "
                       << errors[1] << ", " << errors[2] << " -> orders "
                       << p1 << ", " << p2 << " (expected " << use.expected
                       << ")");
        REQUIRE(errors[0] > 0.0);
        REQUIRE(p1 > use.expected - 0.25);
        REQUIRE(p2 > use.expected - 0.25);
    }
}

TEST_CASE("Transient heat: the adaptive controller earns its accuracy per step",
    "[app][transient]")
{
    // A solution that rises out of nothing within the first 0.02 s and is
    // flat after that, T = t / (t + tau): the steps a uniform scheme has to
    // spend everywhere to resolve the rise are the ones the controller can
    // spend on the rise alone. The solution is uniform in space, so what is
    // compared is the time discretization and nothing else.
    const double tau = 0.02, t_end = 1.0;
    const Manufactured problem {
        "t/(t+" + std::to_string(tau) + ")",
        std::to_string(tau) + "/((t+" + std::to_string(tau) + ")*(t+"
            + std::to_string(tau) + "))",
        [tau](double t) { return t / (t + tau); }};

    const RunResult adaptive = run_adaptive(problem, 1e-3, t_end);
    REQUIRE(adaptive.steps > 0);
    // The rise is steeper than the first step of the run, so the controller
    // has to reject one: the error test is what sets the step here.
    REQUIRE(adaptive.rejected > 0);

    // The tolerance bounds the error of a step, so the worst error over the
    // levels is of the size of the tolerance — a small multiple of it, the
    // local errors accumulating over the levels — and not a fraction of it.
    INFO("adaptive: " << adaptive.steps << " steps (" << adaptive.rejected
                      << " rejected), worst error " << adaptive.worst_error);
    REQUIRE(adaptive.worst_error < 1e-2);
    REQUIRE(adaptive.worst_error > 1e-5);

    // The same number of steps spent uniformly is less accurate, and the same
    // accuracy spent uniformly costs more steps. Both are the point of
    // selecting the step at all.
    const RunResult same_cost
        = run_uniform(problem, t_end / adaptive.steps, t_end);
    INFO("uniform over " << same_cost.steps << " steps: worst error "
                         << same_cost.worst_error);
    REQUIRE(same_cost.steps <= adaptive.steps + 1);
    REQUIRE(adaptive.worst_error < same_cost.worst_error);

    int cheapest = 0;
    for (const int count : {20, 40, 80, 160, 320, 640, 1280, 2560}) {
        const RunResult run = run_uniform(problem, t_end / count, t_end);
        if (run.worst_error <= adaptive.worst_error) {
            cheapest = run.steps;
            break;
        }
    }
    INFO("the cheapest uniform run as accurate as the adaptive one takes "
         << cheapest << " steps");
    REQUIRE(cheapest > 0);
    REQUIRE(cheapest > adaptive.steps);
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

        TimeStepper stepper(*solver, TimeSettings {"bdf2", false, 1e-3});
        stepper.start(0.0);
        for (int n = 1; n <= 20; ++n)
            stepper.step(0.05 * n, 2);
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
