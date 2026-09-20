// hellofem::app — case scheduler tests: field registration, model-driven runs
// SPDX-License-Identifier: MIT

#include "case_scheduler.h"
#include "defaults.h"
#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "fixture.h"
#include "physics_field.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using Catch::Approx;
using namespace hellofem::app;

namespace {

    /// A mesh in the shape the case driver loads it in.
    LoadedMesh loaded(const test::BoxFixture& box)
    {
        LoadedMesh lm;
        lm.mesh = box.mesh;
        lm.cell_tags = box.cells;
        lm.facet_tags = box.boundary;
        lm.num_domains = 1;
        lm.num_boundaries = 6;
        return lm;
    }

    /// The model of the manufactured solution T(x,t) = t^3: uniform in
    /// space, so the spatial discretization is exact and the exported values
    /// measure the time scheme alone. k = rho cp = 1, the source is 3 t^2 and
    /// both x faces carry T = t^3.
    ModelScript manufactured_heat_model(int steps)
    {
        ModelScript model;
        model.name = "manufactured";
        model.materials.push_back(Material {"mat1", {1}, {}, 3,
            {{"thermalconductivity", "1"},
                {"density", "1"},
                {"heatcapacity", "1"}}});
        Physics heat;
        heat.tag = "ht";
        heat.type = "HeatTransfer";
        heat.features = {
            {"temp1", "TemperatureBoundary", {1}, {{"T0", "t*t*t"}}},
            {"temp2", "TemperatureBoundary", {2}, {{"T0", "t*t*t"}}},
            {"hs1", "HeatSource", {1}, {{"Q0", "3*t*t"}}},
            {"init1", "", {}, {{"Tinit", "0"}}},
        };
        model.physics.push_back(std::move(heat));
        model.study.transient = true;
        for (int n = 0; n <= steps; ++n)
            model.study.times.push_back(static_cast<double>(n) / steps);
        model.export_config.expressions = {"T"};
        return model;
    }

    /// The column-header line of a case result file.
    std::string result_header(const std::filesystem::path& path)
    {
        std::ifstream in(path);
        std::string line;
        while (std::getline(in, line))
            if (line.rfind("% x", 0) == 0)
                return line;
        return {};
    }

    /// The times of the stored levels, read off a result header.
    std::vector<double> result_times(const std::string& header)
    {
        std::vector<double> times;
        const std::string token = "@ t=";
        for (std::size_t at = header.find(token); at != std::string::npos;
            at = header.find(token, at + token.size()))
            times.push_back(std::stod(header.substr(at + token.size())));
        return times;
    }

    /// The values of every data row: three coordinates, then one value per
    /// exported column.
    std::vector<std::vector<double>> result_rows(const std::filesystem::path& path)
    {
        std::ifstream in(path);
        std::string line;
        std::vector<std::vector<double>> rows;
        while (std::getline(in, line)) {
            if (line.empty() or line[0] == '%')
                continue;
            std::istringstream row(line);
            std::vector<double> values;
            double value = 0.0;
            while (row >> value)
                values.push_back(value);
            rows.push_back(std::move(values));
        }
        return rows;
    }

    std::size_t count_of(const std::string& text, const std::string& token)
    {
        std::size_t count = 0;
        for (std::size_t at = text.find(token); at != std::string::npos;
            at = text.find(token, at + token.size()))
            ++count;
        return count;
    }

    std::filesystem::path scratch_file(const std::string& name)
    {
        return std::filesystem::temp_directory_path() / name;
    }

} // namespace

TEST_CASE("FieldKind registration: the shipped fields register themselves",
    "[app][scheduler]")
{
    // Every physics registers its kind at program start-up, from its own
    // translation unit — nothing lists them (see `FieldRegistration`).
    for (const char* type : {"ConductiveMedia", "HeatTransfer", "SolidMechanics"})
        REQUIRE(field_kind(type) != nullptr);
    REQUIRE(field_kind("FluidFlow") == nullptr);
}

TEST_CASE("CaseScheduler: a physics without a registered field is an error",
    "[app][scheduler]")
{
    ModelScript model;
    model.physics.push_back(Physics {"ff", "FluidFlow", {}, {}});
    auto box = test::make_box_fixture({0, 0, 0}, {1, 1, 1}, {1, 1, 1});
    REQUIRE_THROWS(CaseScheduler(model, loaded(box), TimeSettings {}));
}

TEST_CASE("A study takes the tightest time tolerance its physics recommend",
    "[app][scheduler]")
{
    // The tolerance a study step that leaves the accuracy to the physics takes
    // is the interfaces' own, and a study over several of them takes the
    // tightest. Measured on the reference by reading its time solver node's
    // `rtol` back: heat transfer and electric currents alone are held to 1e-2,
    // the structural interface to 1e-3, and every combination of them to the
    // tightest of its interfaces (see `defaults.h`).
    const Physics heat {"ht", "HeatTransfer", {}, {}};
    const Physics electric {"ec", "ConductiveMedia", {}, {}};
    const Physics solid {"solid", "SolidMechanics", {}, {}};

    REQUIRE(study_time_tolerance({heat}) == heat_time_tolerance);
    REQUIRE(study_time_tolerance({electric}) == electric_time_tolerance);
    REQUIRE(study_time_tolerance({solid}) == solid_time_tolerance);

    // The combinations the app's own cases are: a heat-transfer-only transient,
    // and the electric-thermal-structural set of the busbar and the packages.
    REQUIRE(study_time_tolerance({heat}) == 1e-2);
    REQUIRE(study_time_tolerance({electric, heat, solid}) == 1e-3);
    REQUIRE(study_time_tolerance({heat, solid}) == 1e-3);
    REQUIRE(study_time_tolerance({heat, electric}) == 1e-2);

    // An interface with no registered field has no tolerance to recommend, and
    // a study of no physics has none either.
    REQUIRE_THROWS(study_time_tolerance({}));
    REQUIRE_THROWS(study_time_tolerance({Physics {"ff", "FluidFlow", {}, {}}}));
}

TEST_CASE("A heat-only study is held to the physics' tolerance, not the tightest",
    "[app][scheduler]")
{
    // The step count of a heat-transfer-only transient is the reference's own
    // only while the study takes the tolerance that interface recommends. The
    // app-wide 1e-3 it used to carry halved the steps: measured on
    // TBlockNonlinTransient, 46 steps where its reference took 24 (14.4 s
    // against 5 s), and 25 steps at 1e-2. This holds the resolution to that
    // direction, on the same manufactured problem at both tolerances.
    const int outputs = 4;
    auto box = test::make_box_fixture({0, 0, 0}, {1, 0.2, 0.2}, {4, 1, 1});
    const ModelScript model = manufactured_heat_model(outputs);

    int runs = 0;
    const auto steps_at = [&](double tolerance) {
        TimeSettings settings;
        settings.tolerance = tolerance;
        settings.store = StoreMode::steps;
        CaseScheduler scheduler(model, loaded(box), settings);
        scheduler.run();
        const auto path = scratch_file(
            "hellofem_tolerance_result_" + std::to_string(++runs) + ".txt");
        scheduler.export_result(path.string());
        return result_times(result_header(path)).size();
    };

    REQUIRE(study_time_tolerance(model.physics) == heat_time_tolerance);
    const std::size_t recommended = steps_at(study_time_tolerance(model.physics));
    const std::size_t tightest = steps_at(solid_time_tolerance);
    INFO("steps at the physics' tolerance = " << recommended
                                              << ", at the tightest = " << tightest);
    REQUIRE(recommended < tightest);
}

TEST_CASE("CaseScheduler: the model's physics drives the solved fields",
    "[app][scheduler]")
{
    // A transient heat model and no electric physics: the scheduler must
    // build the heat field alone, step it, and export exactly the variable
    // that field provides — a column per output time, of the manufactured
    // solution of that time. The steps are the solver's own, and the output
    // times a step stepped over are reported by interpolating it there.
    const int steps = 20;
    auto box = test::make_box_fixture({0, 0, 0}, {1, 0.2, 0.2}, {4, 1, 1});

    {
        CaseScheduler scheduler(manufactured_heat_model(steps), loaded(box),
            TimeSettings {});
        scheduler.run();

        const auto path = scratch_file("hellofem_scheduler_result.txt");
        scheduler.export_result(path.string());

        const std::string header = result_header(path);
        REQUIRE(count_of(header, "T (K)") == static_cast<std::size_t>(steps) + 1);
        REQUIRE(header.find("V (V)") == std::string::npos);
        REQUIRE(header.find("T (K) @ t=1  ") != std::string::npos);

        // Every vertex of every level holds the manufactured solution
        // T = t^3. The Dirichlet vertices carry its data alone; the interior
        // vertices carry the time scheme's own error, so the maximum below is
        // that error and nothing else.
        const std::vector<std::vector<double>> rows = result_rows(path);
        REQUIRE(rows.size() > 0);
        double max_err = 0.0;
        for (const std::vector<double>& row : rows) {
            REQUIRE(row.size() == 3 + static_cast<std::size_t>(steps) + 1);
            for (int n = 0; n <= steps; ++n) {
                const double exact = std::pow(static_cast<double>(n) / steps, 3.0);
                max_err = std::max(
                    max_err, std::abs(row[static_cast<std::size_t>(3 + n)] - exact));
            }
        }
        INFO("steps selected by the error test: manufactured T = t^3, max error "
            << "over the vertices and levels = " << max_err);
        REQUIRE(max_err < 5e-3);
        // A field that merely holds the boundary data would be exact: the
        // interior has to be solved, and no scheme of the app is exact here.
        REQUIRE(max_err > 1e-6);

        // The first level is the initial value of the model, not a solve.
        REQUIRE(rows.front()[3] == Approx(0.0).margin(1e-12));
        REQUIRE(rows.front()[3 + static_cast<std::size_t>(steps)]
            == Approx(1.0).margin(5e-3));
    }
}

TEST_CASE("CaseScheduler: the step modes place the steps, the store mode the result",
    "[app][scheduler]")
{
    // The same manufactured solution over four output times, with the result
    // stored at the solver's own steps: what the export carries is then the
    // step sequence itself, which is what the step modes are about — a mode
    // that holds the steps to the output times, one that forces a step into
    // every subinterval between them, and one that takes a step of the
    // model's own.
    const int outputs = 4;
    const double t_end = 1.0;
    auto box = test::make_box_fixture({0, 0, 0}, {1, 0.2, 0.2}, {4, 1, 1});

    int runs = 0;
    const auto steps_of = [&](TimeSettings settings) {
        settings.store = StoreMode::steps;
        CaseScheduler scheduler(
            manufactured_heat_model(outputs), loaded(box), settings);
        scheduler.run();
        const auto path = scratch_file(
            "hellofem_step_mode_result_" + std::to_string(++runs) + ".txt");
        scheduler.export_result(path.string());
        return result_times(result_header(path));
    };

    // Free: the solver's own steps, which are not the output times.
    {
        const std::vector<double> times = steps_of(TimeSettings {});
        INFO("free: " << times.size() << " steps, last " << times.back());
        REQUIRE(times.size() > static_cast<std::size_t>(outputs) + 1);
    }

    // The end time: the stepping may pass the last output time, whose solution
    // is then interpolated from the steps around it, and with that
    // interpolation off it stops at the last time instead. A manual step that
    // does not divide the span tells the two apart.
    {
        TimeSettings passing;
        passing.steps = StepsMode::manual;
        passing.manual_step = 0.3;
        const std::vector<double> times = steps_of(passing);
        INFO("end time interpolated: last " << times.back() << " of "
                                            << times.size() << " steps");
        REQUIRE(times.back() > t_end);

        TimeSettings stopped = passing;
        stopped.interpolate_end_time = false;
        const std::vector<double> bounded = steps_of(stopped);
        INFO("end time not interpolated: last " << bounded.back());
        REQUIRE(bounded.back() == Approx(t_end));
    }

    // Strict: every step ends at an output time, so each of them is a step.
    {
        TimeSettings settings;
        settings.steps = StepsMode::strict;
        const std::vector<double> times = steps_of(settings);
        for (int n = 0; n <= outputs; ++n) {
            const double output = static_cast<double>(n) / outputs;
            REQUIRE(std::any_of(times.begin(), times.end(), [&](double t) {
                return std::abs(t - output) < 1e-12;
            }));
        }
    }

    // Intermediate: every subinterval of the output times holds a step.
    {
        TimeSettings settings;
        settings.steps = StepsMode::intermediate;
        const std::vector<double> times = steps_of(settings);
        for (int n = 0; n < outputs; ++n) {
            const double lower = static_cast<double>(n) / outputs;
            const double upper = static_cast<double>(n + 1) / outputs;
            REQUIRE(std::any_of(times.begin(), times.end(), [&](double t) {
                return t > lower and t < upper;
            }));
        }
    }

    // Manual: the step is the model's own, so the steps are exactly the ones
    // it states — the time-step error test does not move them.
    {
        TimeSettings settings;
        settings.steps = StepsMode::manual;
        settings.manual_step = 1.0 / 8.0;
        const std::vector<double> times = steps_of(settings);
        INFO("manual: " << times.size() << " steps, spacing "
                        << times[1] - times[0]);
        REQUIRE(times.size() == 9); // the initial state and eight steps
        for (std::size_t n = 1; n < times.size(); ++n)
            REQUIRE(times[n] - times[n - 1] == Approx(0.125).margin(1e-9));
    }

    // Closest: one level per output time, taken from the step nearest it.
    {
        TimeSettings settings;
        settings.store = StoreMode::closest;
        CaseScheduler scheduler(
            manufactured_heat_model(outputs), loaded(box), settings);
        scheduler.run();
        const auto path = scratch_file("hellofem_closest_result.txt");
        scheduler.export_result(path.string());
        const std::vector<double> times = result_times(result_header(path));
        REQUIRE(times.size() == static_cast<std::size_t>(outputs) + 1);
        for (int n = 0; n <= outputs; ++n) {
            const double output = static_cast<double>(n) / outputs;
            REQUIRE(std::abs(times[static_cast<std::size_t>(n)] - output)
                <= max_step_fraction);
        }
    }
}

TEST_CASE("CaseScheduler: an export expression no field provides is an error",
    "[app][scheduler]")
{
    // The model's Data export names the quantities it writes. One that no
    // field provides is refused, not written as a column of zeros: a zero
    // column reads as a solved field that happens to be uniformly zero.
    ModelScript model = manufactured_heat_model(2);
    model.export_config.expressions = {"T", "V"};
    auto box = test::make_box_fixture({0, 0, 0}, {1, 0.2, 0.2}, {2, 1, 1});
    REQUIRE_THROWS_AS(CaseScheduler(model, loaded(box), TimeSettings {}),
        std::runtime_error);
}

TEST_CASE("CaseScheduler: a physics feature the app does not solve is an error",
    "[app][scheduler]")
{
    // A radiation boundary is a condition of another kind. A model carrying
    // one is a model the app cannot solve, and solving it without that
    // boundary would report a field of the wrong problem as its result.
    ModelScript model = manufactured_heat_model(2);
    model.physics.front().features.push_back(PhysicsFeature {
        "rad1", "RadiationBoundary", {1}, {{"epsilon_rad", "0.9"}}});
    auto box = test::make_box_fixture({0, 0, 0}, {1, 0.2, 0.2}, {2, 1, 1});
    REQUIRE_THROWS_AS(CaseScheduler(model, loaded(box), TimeSettings {}),
        std::runtime_error);
}

TEST_CASE("CaseScheduler: a length-valued export follows the geometry's unit",
    "[app][scheduler]")
{
    // A bar held at 393.15 K against a 293.15 K strain reference expands by
    // alpha*DT*L = 1e-5 * 100 * 1 = 1e-3 in SI. COMSOL states a displacement
    // in the geometry's own length unit — its export writes `solid.disp (mm)`
    // for a millimetre model — so the same model written in millimetres has
    // to export the same physical displacement as 1.0, and the column has to
    // say which unit it is in. A solver that works in SI and exports the
    // metre value under a millimetre header reports a thousandth of the
    // displacement its reference does.
    ModelScript model;
    model.name = "expanded_bar";
    model.length_unit = "mm";
    model.materials.push_back(Material {"mat1", {1}, {}, 3,
        {{"thermalconductivity", "1"},
            {"density", "1"},
            {"heatcapacity", "1"},
            {"thermalexpansioncoefficient", "1e-5"},
            {"E", "200e9"},
            {"nu", "0.3"}}});
    Physics heat;
    heat.tag = "ht";
    heat.type = "HeatTransfer";
    heat.features = {
        {"temp1", "TemperatureBoundary", {1}, {{"T0", "393.15"}}},
        {"temp2", "TemperatureBoundary", {2}, {{"T0", "393.15"}}},
        {"init1", "", {}, {{"Tinit", "393.15"}}},
    };
    Physics solid;
    solid.tag = "solid";
    solid.type = "SolidMechanics";
    solid.features = {{"fix1", "Fixed", {1}, {}}};
    model.physics = {heat, solid};
    model.couplings.push_back(MultiphysicsCoupling {"te1", "ThermalExpansion", {1},
        {{"minput_strainreferencetemperature", "293.15"}}});
    model.export_config.expressions = {"solid.disp"};

    auto box = test::make_box_fixture({0, 0, 0}, {1, 0.3, 0.3}, {10, 3, 3});
    CaseScheduler scheduler(model, loaded(box), TimeSettings {});
    scheduler.run();
    const auto path = scratch_file("hellofem_length_unit_result.txt");
    scheduler.export_result(path.string());

    const std::string header = result_header(path);
    REQUIRE(header.find("solid.disp (mm)") != std::string::npos);

    // The x+ face's own displacement: the bar's free end, where the
    // expansion is not held back by the clamped face's lateral constraint.
    const std::vector<std::vector<double>> rows = result_rows(path);
    REQUIRE(rows.size() > 0);
    double ux_max = 0.0;
    for (const std::vector<double>& row : rows) {
        REQUIRE(row.size() == 4);
        if (std::abs(row[0] - 1000.0) < 1e-9) // x = 1 m, written in mm
            ux_max = std::max(ux_max, row[3]);
    }
    INFO("exported solid.disp at the bar's end = " << ux_max
                                                   << " mm, against alpha*DT*L = 1.0 mm");
    REQUIRE(ux_max > 0.9);
    REQUIRE(ux_max < 1.1);
}
