// hellofem::app — COMSOL model-script parsing tests: unit literals,
// muparser expressions and the clean-Java model extractor.
// SPDX-License-Identifier: MIT

#include "Expression.h"
#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "java_parser.h"
#include "units.h"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <set>
#include <unordered_map>

using Catch::Approx;
using hellofem::app::Expression;
using hellofem::app::parse_model_java;
using hellofem::app::parse_si;
using hellofem::app::parse_unit;

TEST_CASE("parse_unit converts to SI", "[app][units]")
{
    REQUIRE(parse_unit("m") == Approx(1.0));
    REQUIRE(parse_unit("cm") == Approx(1e-2));
    REQUIRE(parse_unit("mm") == Approx(1e-3));
    REQUIRE(parse_unit("GPa") == Approx(1e9));
    REQUIRE(parse_unit("MPa") == Approx(1e6));
    REQUIRE(parse_unit("mV") == Approx(1e-3));
    REQUIRE(parse_unit("W/(m*K)") == Approx(1.0));
    REQUIRE(parse_unit("kg/m^3") == Approx(1.0));
    REQUIRE(parse_unit("1/K") == Approx(1.0));
}

TEST_CASE("parse_si handles bare numbers and unit literals", "[app][units]")
{
    REQUIRE(parse_si("0.006") == Approx(0.006));
    REQUIRE(parse_si("20[mV]") == Approx(0.02));
    REQUIRE(parse_si("5[W/m^2/K]") == Approx(5.0));
    REQUIRE(parse_si("110[GPa]") == Approx(1.1e11));
    REQUIRE(parse_si("293.15[K]") == Approx(293.15));
    REQUIRE(parse_si("7.407e5[S/m]") == Approx(740700.0));
}

TEST_CASE("Expression evaluates params and coordinates", "[app][expr]")
{
    double Vtot = 0.02;
    std::unordered_map<std::string, double*> vars;
    vars["Vtot"] = &Vtot;
    Expression e;
    e.parse("Vtot * x / 0.1", vars);
    REQUIRE(e.eval(0.05, 0, 0, 0) == Approx(0.01));
    REQUIRE(e.eval(0.1, 0, 0, 0) == Approx(0.02));

    // x/y/z/t are built-in.
    Expression e2;
    e2.parse("x*y + z*t", vars);
    REQUIRE(e2.eval(2, 3, 4, 5) == Approx(26.0));
}

TEST_CASE("Expression normalizes unit literals", "[app][expr]")
{
    std::unordered_map<std::string, double*> vars;
    Expression e;
    e.parse("20[mV]", vars);
    REQUIRE(e.eval(0, 0, 0, 0) == Approx(0.02));

    // Non-numeric expression referencing params is left to muparser.
    double htc = 5.0;
    double T = 303.15;
    vars["htc"] = &htc;
    vars["T"] = &T;
    Expression e2;
    e2.parse("htc * (T - 293.15)", vars);
    REQUIRE(e2.eval(0, 0, 0, 0) == Approx(50.0));

    // A unit literal on anything but a number is not normalized: the text
    // stays as it is and muparser is the one to reject it. COMSOL scripts
    // reference a named parameter for a value that carries a unit.
    Expression e3;
    REQUIRE_THROWS(e3.parse("T[K]", vars));
}

TEST_CASE("Expression reports used variables", "[app][expr]")
{
    double k = 2.0;
    std::unordered_map<std::string, double*> vars;
    vars["k"] = &k;
    Expression e;
    e.parse("k * sin(x) + y", vars);
    auto vs = e.variables();
    REQUIRE(std::find(vs.begin(), vs.end(), "k") != vs.end());
    REQUIRE(std::find(vs.begin(), vs.end(), "x") != vs.end());
}

namespace {
    /// Write a small clean-Java model to a temp file and return its path.
    std::filesystem::path write_model(const std::string& src)
    {
        const auto path = std::filesystem::temp_directory_path() / "hellofem_model.java";
        std::ofstream out(path);
        out << src;
        return path;
    }

    const std::string sample = R"JAVA(
/* Clean model exported by COMSOL. */
import com.comsol.model.*;
import com.comsol.model.util.*;

public class sample_model {

  public static void main(String[] args) {
    Model model = ModelUtil.create("Model");
    model.modelPath("C:\\cases");

    model.param().set("L", "9[cm]", "length");
    model.param().set("Vtot", "20[mV]", "voltage");
    model.param().set("htc", "5[W/(m^2*K)]", "convection");
    model.param().set("dtout", "0.5[s]", "output interval");
    model.param().set("tend", "2[s]", "end time");

    String comp = "comp1";
    model.component().create(comp, true);

    model.component("comp1").geom().create("geom1", 3);
    model.component("comp1").geom("geom1").create("blk1", "Block");
    model.component("comp1").geom("geom1").feature("blk1").set("size", new String[]{"L","L","L"});
    model.component("comp1").geom("geom1").run();

    model.component("comp1").material().create("mat1", "Common");
    model.component("comp1").material("mat1").selection().set(new int[]{1});
    model.component("comp1").material("mat1").propertyGroup("def")
        .set("electricconductivity", new String[][]{{"sig"}});
    model.component("comp1").material("mat1").propertyGroup("Enu")
        .set("E", new String[][]{{"110[GPa]"}});

    model.component("comp1").physics().create("ec", "ConductiveMedia", "geom1");
    model.component("comp1").physics("ec").create("term1", "Terminal", 2);
    model.component("comp1").physics("ec").feature("term1").selection().set(new int[]{1});
    model.component("comp1").physics("ec").feature("term1").set("TerminalType", "Voltage");
    model.component("comp1").physics("ec").feature("term1").set("V0", "Vtot");
    model.component("comp1").physics("ec").create("gnd1", "Ground", 2);
    model.component("comp1").physics("ec").feature("gnd1").selection().set(new int[]{2});

    model.component("comp1").multiphysics().create("emh1", "ElectromagneticHeating");
    model.component("comp1").multiphysics("emh1").set("EMHeat_physics", "ec");

    model.component("comp1").mesh().create("mesh1");
    model.component("comp1").mesh("mesh1").autoMeshSize(2);
    model.component("comp1").mesh("mesh1").run();

    model.study().create("std1");
    model.study("std1").create("time", "Transient");
    model.study("std1").feature("time").set("tlist", "range(0,dtout,tend)");
    model.study("std1").createAutoSequences("time");
    model.study("std1").run();

    model.result().export().create("data1", "Data");
    model.result().export("data1").set("data", "dset1");
    model.result().export("data1").set("filename", "result.txt");
    model.result().export("data1").set("expr", new String[]{"V", "T", "solid.disp"});
    model.result().export("data1").run();

    model.save("model.mph");
  }
}
)JAVA";

    /// Write the sample model with `extra` statements inserted into its
    /// transient study step, before the solver sequence is created. A missing
    /// anchor throws, so a test cannot pass on an insertion that never
    /// happened.
    std::filesystem::path write_model_with_study(const std::string& extra)
    {
        const std::string anchor
            = "    model.study(\"std1\").createAutoSequences(\"time\");";
        std::string text = sample;
        text.insert(text.find(anchor), extra);
        return write_model(text);
    }
} // namespace

TEST_CASE("parse_model_java extracts params/materials/physics", "[app][java]")
{
    auto path = write_model(sample);
    auto model = parse_model_java(path);

    REQUIRE(model.parameters.size() == 5);
    REQUIRE(model.parameters[0].name == "L");
    REQUIRE(model.parameters[0].si == Catch::Approx(0.09));
    REQUIRE(model.parameters[1].name == "Vtot");
    REQUIRE(model.parameters[1].si == Catch::Approx(0.02));

    REQUIRE(model.materials.size() == 1);
    REQUIRE(model.materials[0].tag == "mat1");
    REQUIRE(model.materials[0].domains == std::set<int> {1});
    REQUIRE(model.materials[0].properties.size() == 2);
    REQUIRE(model.materials[0].properties[0].name == "electricconductivity");

    REQUIRE(model.physics.size() == 1);
    REQUIRE(model.physics[0].type == "ConductiveMedia");
    REQUIRE(model.physics[0].features.size() == 2);
    const auto& term = model.physics[0].features[0];
    REQUIRE(term.type == "Terminal");
    REQUIRE(term.selection == std::set<int> {1});
    REQUIRE(term.properties.at("TerminalType") == "Voltage");
    REQUIRE(term.properties.at("V0") == "Vtot");

    REQUIRE(model.couplings.size() == 1);
    REQUIRE(model.couplings[0].type == "ElectromagneticHeating");

    REQUIRE(model.study.transient);
    REQUIRE(model.study.times.size() == 5);
    REQUIRE(model.study.times[1] == Catch::Approx(0.5));
    REQUIRE(model.study.times.back() == Catch::Approx(2.0));

    REQUIRE(model.export_config.expressions.size() == 3);
    REQUIRE(model.export_config.expressions[2] == "solid.disp");
}

TEST_CASE("a physics-controlled study sets no time tolerance", "[app][java]")
{
    // COMSOL holds a study the model does not give a tolerance to the one its
    // physics recommends. Such a study carries no `rtol` at all, and one whose
    // `usertol` is off carries an `rtol` that does not apply.
    auto plain = parse_model_java(write_model(sample));
    REQUIRE(plain.study.transient);
    REQUIRE_FALSE(plain.study.tolerance.has_value());

    auto physics_controlled = parse_model_java(write_model_with_study(
        "    model.study(\"std1\").feature(\"time\").set(\"usertol\", \"off\");\n"
        "    model.study(\"std1\").feature(\"time\").set(\"rtol\", \"1e-6\");\n"));
    REQUIRE_FALSE(physics_controlled.study.tolerance.has_value());
}

TEST_CASE("parse_model_java reads a user-set time tolerance", "[app][java]")
{
    auto model = parse_model_java(write_model_with_study(
        "    model.study(\"std1\").feature(\"time\").set(\"usertol\", \"on\");\n"
        "    model.study(\"std1\").feature(\"time\").set(\"rtol\", \"1e-6\");\n"));
    REQUIRE(model.study.transient);
    REQUIRE(model.study.tolerance.has_value());
    REQUIRE(*model.study.tolerance == Catch::Approx(1e-6));
}

TEST_CASE("the study tolerance may name a parameter", "[app][java]")
{
    // A model may state its tolerance as a parameter, so the value resolves
    // once every parameter is known — the same way the time list does.
    auto model = parse_model_java(write_model_with_study(
        "    model.param().set(\"tol_rel\", \"1e-5\", \"step tolerance\");\n"
        "    model.study(\"std1\").feature(\"time\").set(\"usertol\", \"on\");\n"
        "    model.study(\"std1\").feature(\"time\").set(\"rtol\", \"tol_rel\");\n"));
    REQUIRE(model.study.tolerance.has_value());
    REQUIRE(*model.study.tolerance == Catch::Approx(1e-5));
}
