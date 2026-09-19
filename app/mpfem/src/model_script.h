// hellofem::app — model definition extracted from a clean COMSOL Java script
// SPDX-License-Identifier: MIT
#pragma once

#include "units.h"

#include <map>
#include <optional>
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
        /// 1-based COMSOL boundary ids, for a material COMSOL selected on a
        /// *surface*: the thin layers of a case take their properties from it
        /// rather than from the domain's material.
        std::set<int> boundaries;
        /// Dimension of the selection the material was assigned on: 3 for
        /// domains, 2 for boundaries. COMSOL states it in one statement
        /// (`selection().geom(g, dim)`) and the entities in the next, so the
        /// parser carries it across the two.
        int selection_dim = 3;
        std::vector<MaterialProperty> properties;

        /// The property `name` of this material, if it defines one.
        const MaterialProperty* property(std::string_view name) const;
    };

    /// A boundary-condition / feature definition on a physics interface.
    struct PhysicsFeature {
        std::string tag; // feature tag ("term1", "gnd1", ...)
        /// Feature type ("Terminal", "Ground", "HeatFluxBoundary", "Fixed",
        /// ...). Empty for a feature the model never created: a `.feature(tag)`
        /// the script only sets properties on is one of the interface's own
        /// defaults, which COMSOL creates with the interface itself.
        std::string type;
        std::set<int> selection; // boundary/domain ids (1-based COMSOL)
        std::map<std::string, std::string> properties;

        /// The property `key`, or an error naming the feature. A property the
        /// app reads and the model does not state is a feature the app cannot
        /// solve, not one to fall back on a default for.
        const std::string& required(std::string_view key) const;
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
        /// Relative tolerance the study holds its steps to, when the model
        /// sets one — the model's own accuracy for the time discretization.
        /// A study the model leaves physics controlled sets none, and the app
        /// uses COMSOL's own value (`defaults.h`).
        std::optional<double> tolerance;
        /// The study step's `usertol` and `rtol`, resolved into `tolerance`
        /// once every model parameter is known. The two are kept apart because
        /// `rtol` applies only where `usertol` turns the physics control off.
        bool user_tolerance = false;
        std::string tolerance_expr;
    };

    /// Result export configuration (expressions to compare).
    struct ExportConfig {
        std::vector<std::string> expressions; // "V", "T", "solid.disp", ...
    };

    /// Fully-parsed model definition from the clean Java script.
    struct ModelScript {
        std::string name;
        /// Length unit of the geometry, from the model's
        /// `lengthUnit("mm")` (the metre when the model states none).
        ///
        /// The geometry — and the mesh COMSOL exports from it, and the
        /// coordinates of its data export — is in that unit, while every
        /// material law, parameter and feature the model states is in SI.
        /// The app solves in SI, so the mesh is brought in with
        /// `length_scale()` and the exported coordinates are written back in
        /// this unit, which is what COMSOL's own export does.
        std::string length_unit = "m";

        /// SI metres per `length_unit`.
        double length_scale() const { return parse_unit(length_unit); }

        std::vector<Parameter> parameters;
        std::vector<Material> materials;
        std::vector<Physics> physics;
        std::vector<MultiphysicsCoupling> couplings;
        StudyConfig study;
        ExportConfig export_config;

        const Material* material_on_domain(int domain) const;

        /// The material COMSOL applies on `boundary`, i.e. the material of
        /// its boundary (surface) selection.
        ///
        /// A domain or boundary two materials both select goes to the one
        /// COMSOL lists last, which is its own resolution rule for an
        /// overlap.
        const Material* material_on_boundary(int boundary) const;
    };

} // namespace hellofem::app
