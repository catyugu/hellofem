// hellofem::app — model definition extracted from a clean COMSOL Java script
// SPDX-License-Identifier: MIT
#pragma once

#include <map>
#include <set>
#include <string>
#include <vector>

namespace hellofem::app {

    /// A named scalar parameter (e.g. `model.param().set("Vtot","20[mV]")`).
    struct Parameter {
        std::string name;
        std::string value; // COMSOL expression with optional unit literal
        double si = 0.0; // evaluated SI value (parse_si)
    };

    /// A material property set on a domain selection.
    struct MaterialProperty {
        std::string name; // "electricconductivity", "thermalconductivity", ...
        std::string value; // expression string (possibly a 9-entry tensor)
    };

    /// A material and its domain assignment + properties.
    struct Material {
        std::string tag;
        std::string label;
        std::set<int> domains; // 1-based COMSOL domain ids
        std::vector<MaterialProperty> properties;
    };

    /// A boundary-condition / feature definition on a physics interface.
    struct PhysicsFeature {
        std::string tag; // feature tag ("term1", "gnd1", ...)
        std::string type; // "Terminal", "Ground", "HeatFluxBoundary", "Fixed", ...
        std::set<int> selection; // boundary/domain ids (1-based COMSOL)
        std::map<std::string, std::string> properties;
    };

    /// A physics interface (electrostatics/heat/solid).
    struct Physics {
        std::string tag; // "ec", "ht", "solid"
        std::string type; // "ConductiveMedia", "HeatTransfer", "SolidMechanics"
        std::vector<PhysicsFeature> features;
    };

    /// A multiphysics coupling (Joule heating, thermal expansion).
    struct MultiphysicsCoupling {
        std::string tag; // "emh1", "te1"
        std::string type; // "ElectromagneticHeating", "ThermalExpansion"
        std::set<int> domains; // 1-based domain ids
        std::map<std::string, std::string> properties;
    };

    /// Study configuration.
    struct StudyConfig {
        std::string type = "Stationary"; // "Stationary" | "Transient"
        std::vector<double> times; // transient output times (tlist)
        int mesh_refine = 2; // autoMeshSize hint
    };

    /// Result export configuration (expressions to compare).
    struct ExportConfig {
        std::vector<std::string> expressions; // "V", "T", "solid.disp", ...
    };

    /// Fully-parsed model definition from the clean Java script.
    struct ModelScript {
        std::string name;
        std::vector<Parameter> parameters;
        std::vector<Material> materials;
        std::vector<Physics> physics;
        std::vector<MultiphysicsCoupling> couplings;
        StudyConfig study;
        ExportConfig export_config;

        const Parameter* parameter(const std::string& name) const;
        const Material* material_on_domain(int domain) const;
        const Physics* physics_by_type(const std::string& type) const;
        std::vector<const PhysicsFeature*> features(const std::string& type) const;
    };

} // namespace hellofem::app
