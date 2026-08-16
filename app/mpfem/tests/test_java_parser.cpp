// hellofem::app — clean-Java model-script parser tests
// SPDX-License-Identifier: MIT

#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "java_parser.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <set>

using hellofem::app::ModelScript;
using hellofem::app::parse_model_java;

namespace {
    /// Write a small clean-Java model to a temp file and return its path.
    std::filesystem::path write_model(const std::string& src)
    {
        const auto path = std::filesystem::temp_directory_path() / "hellofem_model.java";
        std::ofstream out(path);
        out << src;
        return path;
    }

    const std::string sample = R"(
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
    model.study("std1").create("stat", "Stationary");
    model.study("std1").createAutoSequences("stat");
    model.study("std1").run();

    model.result().export().create("data1", "Data");
    model.result().export("data1").set("data", "dset1");
    model.result().export("data1").set("filename", "result.txt");
    model.result().export("data1").set("expr", new String[]{"V", "T", "solid.disp"});
    model.result().export("data1").run();

    model.save("model.mph");
  }
}
)";
} // namespace

TEST_CASE("parse_model_java extracts params/materials/physics", "[app][java]")
{
    auto path = write_model(sample);
    auto model = parse_model_java(path);

    REQUIRE(model.parameters.size() == 3);
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

    REQUIRE(model.study.type == "Stationary");
    REQUIRE(model.study.mesh_refine == 2);

    REQUIRE(model.export_config.expressions.size() == 3);
    REQUIRE(model.export_config.expressions[2] == "solid.disp");
}
