// hellofem::app — model definition extracted from a clean COMSOL Java script
// SPDX-License-Identifier: MIT
#pragma once

#include <map>
#include <set>
#include <string>
#include <string_view>
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

        /// The value as a scalar expression. COMSOL stores a tensor-valued
        /// property as a space-separated component list; the isotropic case
        /// uses the leading component.
        std::string scalar_value() const;
    };

    /// A material and its domain assignment + properties.
    struct Material {
        std::string tag;
        std::set<int> domains; // 1-based COMSOL domain ids
        std::vector<MaterialProperty> properties;

        /// The property `name` of this material, if it defines one.
        const MaterialProperty* property(std::string_view name) const;
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
        /// Element order per dependent variable, as the model's
        /// `ShapeProperty` sets it (`order_<variable>` -> order). A physics
        /// the model leaves alone is absent, and the app uses COMSOL's own
        /// default (see `defaults.h`).
        std::map<std::string, int> element_order;
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
        /// Time-dependent study (a stationary one otherwise).
        bool transient = false;
        /// Output times of a transient study, resolved from `tlist`.
        std::vector<double> times;
        /// Raw `tlist` expression, resolved into `times` once every model
        /// parameter is known.
        std::string times_expr;
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

        const Material* material_on_domain(int domain) const;
    };

} // namespace hellofem::app
