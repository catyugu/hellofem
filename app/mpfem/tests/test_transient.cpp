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
        /// Its value at t = 0, which the initial state must carry: a solution
        /// that does not start where its boundary data does is an inconsistent
        /// initial condition, and the first step would jump to the boundary
        /// instead of following the solution.
        std::string initial = "0";
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
        solver->add_source({1}, source);
        solver->add_temperature_bc(1, ScalarExpression(problem.exact));
        solver->add_temperature_bc(2, ScalarExpression(problem.exact));
        solver->set_initial_temperature(ScalarExpression(problem.initial));
        solver->apply_initial_condition(0.0);
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
    double manufactured_error(int order, double dt, double t_end)
    {
        const Manufactured problem = cubic_problem();
        auto solver = manufactured_solver(problem);

        TimeSettings settings;
        settings.max_order = order;
        TimeStepper stepper(*solver, settings);
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
        TimeSettings settings;
        settings.max_order = 2;
        auto solver = manufactured_solver(problem);
        TimeStepper stepper(*solver, settings);
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

    /// Integrate `problem` to `t_end` with the case's own step control: the
    /// `BdfController` the scheduler drives, over the one field, with the
    /// initialization step and the initial-step rule it starts the run with.
    RunResult run_adaptive(const Manufactured& problem, double tolerance,
        double t_end, double absolute_factor = 0.1)
    {
        TimeSettings settings;
        settings.tolerance = tolerance;
        settings.absolute_factor = absolute_factor;
        auto solver = manufactured_solver(problem);
        TimeStepper stepper(*solver, settings);
        stepper.start(0.0);

        BdfController control(settings, t_end);
        // The consistent-initialization step, and the derivative the first
        // step's condition is read from.
        const double probe
            = t_end * first_step_fraction * backward_euler_step_fraction;
        stepper.step(probe, 1);
        double t = probe;
        control.start(control.first_step(stepper.derivative_norm()));

        RunResult run;
        while (t < t_end) {
            const double h = std::min(control.step_size(), t_end - t);
            stepper.step(t + h, control.order());
            const StepError error = stepper.error_estimates();
            if (error.at > 1.0) {
                stepper.undo();
                control.reject(error, h);
                ++run.rejected;
                REQUIRE(run.rejected < 1000); // the control has to converge
                continue;
            }
            t += h;
            control.accept(error, h);
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

TEST_CASE("BDF: the weights follow the steps, the controller the estimates",
    "[app][transient]")
{
    const double dt = 0.5;

    TimeWeights w = bdf_weights(1, TimeSteps {dt, dt});
    REQUIRE(w.a.size() == 2);
    REQUIRE(w.a[0] == Approx(2.0));
    REQUIRE(w.a[1] == Approx(-2.0));
    REQUIRE(w.b[0] == Approx(1.0));
    REQUIRE(w.b[1] == Approx(0.0));

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

    // The first step: 0.1% of the span, and the derivative condition
    // DT yp_norm <= 1/2, which is what takes it below that.
    const double span = 600.0;
    // The controller's rules are exercised over the range the reference's own
    // solver runs at (orders 1 to 2). The app's default is 1 (see
    // `TimeSettings`), where the startup phase and the order selection never
    // come into play.
    TimeSettings range;
    range.max_order = 2;
    BdfController first(range, span);
    REQUIRE(first.first_step(0.0) == Approx(0.6));
    REQUIRE(first.first_step(0.1) == Approx(0.6)); // 0.5 / 0.1 = 5 s
    REQUIRE(first.first_step(1.0) == Approx(0.5)); // the condition binds

    // The startup phase, on the reference's own span: the first two steps are
    // the initial step, then every step is taken one order higher and at
    // twice the size until the order reaches its maximum, and the step stops
    // at the largest one. The reference's log reads 0.6, 0.6, 1.2, 2.4, 4.8,
    // 9.6, 19.2, ... s over this span, at orders 1, 1, 2, 2, ... — the same
    // sequence, held twice more there by an estimate that its own weights
    // made larger.
    {
        BdfController control(range, span);
        control.start(control.first_step(0.0));
        std::vector<double> sizes;
        std::vector<int> orders;
        for (int n = 0; n < 11; ++n) {
            sizes.push_back(control.step_size());
            orders.push_back(control.order());
            control.accept(StepError {1e-9, 1e-9, 1e-9}, control.step_size());
        }
        INFO("steps: " << sizes[0] << ", " << sizes[1] << ", " << sizes[2]);
        REQUIRE(sizes
            == std::vector<double> {0.6, 0.6, 1.2, 2.4, 4.8, 9.6, 19.2, 38.4,
                60.0, 60.0, 60.0});
        REQUIRE(orders
            == std::vector<int> {1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2});
    }

    // A controller driven to its steady state: the startup has taken the
    // order to the top of the range and the step to twice the initial one,
    // where the estimates, and not the startup, select what comes next.
    const auto settled = [&]() {
        BdfController control(range, span);
        control.start(control.first_step(0.0));
        for (int n = 0; n < 3; ++n)
            control.accept(StepError {1e-9, 1e-9, 1e-9}, control.step_size());
        return control;
    };
    const double settled_step = 2.4;

    // The deadbeat region: an estimate inside it holds the step where it is,
    // and the step doubles only once the estimate is below 2^-(q+2) of the
    // tolerance — 1/16 at order 2, which is the "16 times smaller" the
    // reference's documentation describes. A controller that grows the step
    // by the asymptotic factor from inside the region takes four times the
    // steps here.
    {
        BdfController held = settled();
        REQUIRE(held.order() == 2);
        REQUIRE(held.step_size() == Approx(settled_step));
        held.accept(StepError {0.2, 0.2, 0.0}, settled_step);
        REQUIRE(held.step_size() == Approx(settled_step)); // the deadbeat region

        BdfController grown = settled();
        grown.accept(StepError {0.03, 0.03, 0.0}, settled_step); // below 1/16
        REQUIRE(grown.step_size() == Approx(2.0 * settled_step));

        BdfController shrunk = settled();
        shrunk.accept(StepError {0.8, 0.8, 0.0}, settled_step);
        // The factor of a shrinking step is the same estimate the growth
        // reads, with the guard the implementation adds to it (1e-4).
        REQUIRE(shrunk.step_size()
            == Approx(settled_step * std::pow(2.0 * 0.8 + 1e-4, -1.0 / 3.0)));
    }

    // A rejected step is retried at the size the asymptotic dependence of the
    // error on the step allows, at most 0.9 and at least 0.25 of it, and at a
    // quarter of the step from the second rejection on.
    {
        BdfController rejected = settled();
        rejected.reject(StepError {4.0, 4.0, 0.0}, settled_step);
        REQUIRE(rejected.step_size()
            == Approx(settled_step * 0.9 * std::pow(8.0, -1.0 / 3.0)));
        rejected.reject(StepError {4.0, 4.0, 0.0}, settled_step);
        REQUIRE(rejected.step_size() == Approx(settled_step * 0.25));
    }

    // The order is reconsidered only once the step size and the order have
    // settled — `q + 2` steps at them — and it is raised when the estimate at
    // the order above is below half the one at the order in use.
    {
        const auto orders_with = [&](double above) {
            BdfController control = settled();
            // The estimate at the order below brings the order back to 1,
            // where the step size then settles.
            control.accept(StepError {0.0, 0.2, 0.2}, control.step_size());
            std::vector<int> seen;
            for (int n = 0; n < 5; ++n) {
                seen.push_back(control.order());
                control.accept(StepError {0.2, 0.2, above}, control.step_size());
            }
            return seen;
        };
        REQUIRE(orders_with(0.2) == std::vector<int> {1, 1, 1, 1, 1});
        REQUIRE(orders_with(1e-6) == std::vector<int> {1, 1, 1, 2, 2});
    }

    // The error weight is `R (absolute_factor + |u|)`. A heat problem with no
    // boundary condition at all keeps the field uniform — the spatial operator
    // annihilates it — so every dof carries the same weight and the two
    // factors can be read off the estimate exactly.
    {
        const auto estimate = [&](double tolerance, double absolute_factor) {
            TimeSettings settings;
            settings.tolerance = tolerance;
            settings.absolute_factor = absolute_factor;
            auto box = test::make_box_fixture({0, 0, 0}, {1, 0.2, 0.2}, {4, 1, 1});
            auto solver = std::make_shared<HeatTransferSolver>(
                box.mesh, box.boundary, box.cells, 1);
            solver->set_conductivity(
                test::constant_property(box.mesh, box.cells, 1.0));
            solver->set_thermal_mass(
                test::constant_property(box.mesh, box.cells, 1.0));
            auto source = std::make_shared<CellProperty>(
                box.mesh, box.cells, no_params());
            source->set_expression(1, "3*t*t");
            solver->add_source({1}, source);
            solver->set_initial_temperature(ScalarExpression(0.0));
            solver->apply_initial_condition(0.0);

            TimeStepper stepper(*solver, settings);
            stepper.start(0.0);
            for (int n = 1; n <= 3; ++n)
                stepper.step(0.1 * n, 1);
            return stepper.error_estimates().at;
        };
        const double at_1e3 = estimate(1e-3, 0.1);
        REQUIRE(at_1e3 > 0.0);
        // Twice the tolerance, half the estimate.
        REQUIRE(estimate(2e-3, 0.1) == Approx(0.5 * at_1e3));
        // Backward Euler from zero with the source 3 t^2 takes the uniform
        // solution to 3 h (t_1^2 + t_2^2 + t_3^2) = 0.042 at the newest
        // level, which is the value the weight of every dof carries: the
        // factor of 1.1 lowers the estimate by (0.1 + u) / (1.1 + u).
        const double u = 0.042;
        REQUIRE(estimate(1e-3, 1.1) == Approx(at_1e3 * (0.1 + u) / (1.1 + u)));
    }

    // The orders above the app's range are refused rather than silently
    // stepped at the highest one it implements.
    REQUIRE_THROWS(BdfController(TimeSettings {1, 3}, span));
    REQUIRE_THROWS(BdfController(TimeSettings {3, 2}, span));
}

TEST_CASE("Transient heat: time scheme order on a manufactured solution", "[app][transient]")
{
    // The time discretization error is measured alone (the manufactured
    // solution is spatially uniform). Backward Euler is first order and BDF2
    // second order; the BDF2 start-up step is first order but its local error
    // is one order higher, so the global order stays two.
    const double t_end = 1.0;
    const std::vector<double> steps {0.05, 0.025, 0.0125};
    struct Use {
        int order;
        double expected;
    };
    for (const Use& use : std::vector<Use> {{1, 1.0}, {2, 2.0}}) {
        std::vector<double> errors;
        for (double dt : steps)
            errors.push_back(manufactured_error(use.order, dt, t_end));
        const double p1 = observed_order(errors[0], errors[1]);
        const double p2 = observed_order(errors[1], errors[2]);
        INFO("order " << use.order << ": errors = " << errors[0] << ", "
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
    // Measured on this problem: the controller rejects nothing. The step it
    // holds inside the deadbeat region is the size the rise is resolved at,
    // and the doubling of the startup phase never overshoots it — where the
    // elementary controller grew every step by 1.25 of what the estimate
    // allowed and had to throw the overshoot away.
    INFO("adaptive: " << adaptive.steps << " steps ("
                      << adaptive.rejected << " rejected)");

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

TEST_CASE("Transient heat: the absolute tolerance keeps a decaying solution moving",
    "[app][transient]")
{
    // T = (1 - t)^3 decays to zero with a vanishing derivative, so towards the
    // end of the run a purely relative criterion has no magnitude left to
    // stand on and would shrink the step to follow the value down. The
    // reference's absolute tolerance `A = R * absolute_factor` is what the
    // weight of a dof with a vanishing value is set from, so the step stays
    // the one the accuracy asks for instead of the one a vanishing value
    // would. Measured below: the factor is what the step count follows.
    const double t_end = 1.0;
    const Manufactured problem {
        "(1-t)*(1-t)*(1-t)", "-3*(1-t)*(1-t)",
        [](double t) { return (1.0 - t) * (1.0 - t) * (1.0 - t); }, "1"};

    const RunResult run = run_adaptive(problem, 1e-3, t_end);
    INFO("decay to zero: " << run.steps << " steps (" << run.rejected
                           << " rejected), worst error " << run.worst_error);
    REQUIRE(run.steps > 0);
    REQUIRE(run.worst_error < 1e-2);

    // A larger absolute tolerance is a larger floor under the weight, so the
    // same run takes fewer steps; the smaller one it is, the closer the
    // criterion is to a purely relative one.
    const RunResult loose = run_adaptive(problem, 1e-3, t_end, 1.0);
    INFO("with an absolute factor of 1: " << loose.steps << " steps");
    REQUIRE(loose.steps < run.steps);
}

TEST_CASE("Transient heat: the initial value is taken at the study's start time",
    "[app][transient]")
{
    // A study whose output times start at t0 > 0 evaluates the model's
    // initial-value expression there, not at zero: a T that follows `t`
    // starts where the study does, and that state is what the scheme reads
    // as the history of its first step.
    auto box = test::make_box_fixture({0, 0, 0}, {1, 0.2, 0.2}, {4, 1, 1});
    auto solver = std::make_shared<HeatTransferSolver>(
        box.mesh, box.boundary, box.cells, 1);
    solver->set_conductivity(test::constant_property(box.mesh, box.cells, 1.0));
    solver->set_thermal_mass(test::constant_property(box.mesh, box.cells, 1.0));
    solver->set_initial_temperature(ScalarExpression("t/2"));

    solver->apply_initial_condition(2.0);

    double worst = 0.0;
    for (double value : solver->solution()->x()->array())
        worst = std::max(worst, std::abs(value - 1.0));
    INFO("initial state deviation from t0 / 2 = " << worst);
    REQUIRE(worst == Approx(0.0).margin(1e-12));
}

TEST_CASE("Transient heat: a solution-independent nonlinear law matches the linear one",
    "[app][transient]")
{
    // k(T) = k0 with alpha_k = 0 is the constant property, whether or not the
    // expression reads the field; the two must reproduce each other.
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
        solver->add_source({1}, source);
        solver->add_temperature_bc(1, ScalarExpression("t*t*t"));
        solver->add_temperature_bc(2, ScalarExpression("t*t*t"));
        solver->set_initial_temperature(ScalarExpression(0.0));
        solver->apply_initial_condition(0.0);

        TimeStepper stepper(*solver, TimeSettings {});
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
