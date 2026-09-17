// hellofem::app — case scheduler tests: field registration, model-driven runs
// SPDX-License-Identifier: MIT

#include "case_scheduler.h"
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
        lm.order = 1;
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
        model.materials.push_back(Material {"mat1", {1},
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
            {"init1", "init1", {}, {{"Tinit", "0"}}},
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
    model.physics.push_back(Physics {"ff", "FluidFlow", {}});
    auto box = test::make_box_fixture({0, 0, 0}, {1, 1, 1}, {1, 1, 1});
    REQUIRE_THROWS(CaseScheduler(model, loaded(box), TimeSettings {}));
}

TEST_CASE("CaseScheduler: the model's physics drives the solved fields",
    "[app][scheduler]")
{
    // A transient heat model and no electric physics: the scheduler must
    // build the heat field alone, step it, and export exactly the variable
    // that field provides — a column per output time, of the manufactured
    // solution of that time. The levels are the model's output times whatever
    // the scheme: the fixed-order one steps to each of them, the adaptive one
    // steps within the intervals and lands on them.
    const int steps = 20;
    auto box = test::make_box_fixture({0, 0, 0}, {1, 0.2, 0.2}, {4, 1, 1});

    for (const TimeSettings& settings :
        {TimeSettings {"bdf2", false, 1e-3}, TimeSettings {"bdf2", true, 1e-3}}) {
        CaseScheduler scheduler(manufactured_heat_model(steps), loaded(box),
            settings);
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
        INFO((settings.adaptive ? "adaptive steps" : "output-time steps")
            << ": manufactured T = t^3, max error over the vertices and levels = "
            << max_err);
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
