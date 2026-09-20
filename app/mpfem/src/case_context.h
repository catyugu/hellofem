// hellofem::app — the model data a physics binds itself against
// SPDX-License-Identifier: MIT
#pragma once

#include "field.h"
#include "mesh_loader.h"
#include "model_script.h"
#include "property.h"

#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>

namespace hellofem::app {

    /// The model data a physics reads while binding itself, plus the solutions
    /// the fields published so far. A material law may read the dependent
    /// variables of the model (a conductivity `k(T)`), which is COMSOL's own
    /// scoping rule: inside a physics-owned expression the dependent variable
    /// wins over a model parameter of the same name.
    ///
    /// The context owns the model and the mesh it was built from: the data a
    /// physics binds itself against — a coefficient reading a material law, a
    /// boundary value — is built from them and outlives the binding call.
    class CaseContext {
    public:
        /// @param[in] model The model, taken by value.
        /// @param[in] mesh The mesh, taken by value.
        /// @param[in] time Time stepping scheme and tolerance of a transient
        /// study (see `TimeSettings`).
        CaseContext(ModelScript model, LoadedMesh mesh, TimeSettings time);

        const ModelScript& model() const { return model_; }
        const LoadedMesh& mesh() const { return mesh_; }
        const TimeSettings& time_settings() const { return time_; }

        /// The element order of a physics interface's dependent variable:
        /// what the model's `ShapeProperty` sets for it, or COMSOL's own
        /// default. The order belongs to the physics, not to the mesh — a
        /// linear geometry carries a quadratic field.
        int element_order(
            const Physics& physics, std::string_view variable) const;

        /// Compile a model expression with the parameters bound by value.
        ScalarExpression expression(std::string_view text) const;

        /// A cell coefficient: `value(material)` gives its expression on the
        /// domains of that material (an empty string leaves a domain without
        /// one, i.e. at the zero of the coefficient).
        std::shared_ptr<DomainProperty> property(
            const std::function<std::string(const Material&)>& value) const;

        /// The material property `name`, on every domain whose material
        /// defines one.
        std::shared_ptr<DomainProperty> material_property(
            std::string_view name) const;

        /// One expression on every domain of the mesh. A cell coefficient
        /// reads its value per cell, so a source on a set of domains is one
        /// integral with one such coefficient.
        std::shared_ptr<DomainProperty> uniform_property(
            std::string_view text) const;

        /// The value of boundary `boundary` (1-based COMSOL id): one model
        /// expression on that boundary. A facet integral reads its coefficient
        /// off the cell next to its facet, so a boundary condition whose value
        /// differs from one boundary to the next is not one integral with one
        /// coefficient but one integral per boundary, each carrying its own.
        std::shared_ptr<FacetProperty> facet_property(
            std::string_view text, int boundary) const;

        /// Publish the solution a physics solves under `symbol`, the COMSOL
        /// dependent-variable name every model expression refers to it by, so
        /// that the coefficients built after this call may read it. A physics
        /// calls this before building its own coefficients, so that a law of
        /// the physics reads the field it solves.
        void bind_solution(std::string_view symbol,
            std::shared_ptr<const fem::Function<double>> solution);

        /// Set the state the rest of the case reads the field at, under the
        /// same symbol: the sampled state of a field the study advances in
        /// time (see `TimeStepper::sample`), and the solution of one it does
        /// not. This is what a coupling of another physics — and the result
        /// export — reads (see `CaseScheduler::bring_to`).
        ///
        /// The two are separate calls because they answer different questions:
        /// a field's own law reads the field it solves, and the rest of the
        /// case reads the state the scheduler brought it to.
        void publish_state(std::string_view symbol,
            std::shared_ptr<const fem::Function<double>> state);

        /// The solution published under `symbol`, or nullptr.
        std::shared_ptr<const fem::Function<double>> solution(
            std::string_view symbol) const;

    private:
        /// A cell coefficient without a model value: zero on every domain,
        /// with every solution published so far bound to it. The public
        /// lookups start from it and fill it per domain.
        std::shared_ptr<DomainProperty> zero_property() const;

        /// The model and the mesh the context owns, and the time stepping of
        /// the study (its tolerance resolved, see `CaseScheduler`).
        ModelScript model_;
        LoadedMesh mesh_;
        TimeSettings time_;
        std::unordered_map<std::string, double> params_;
        std::unordered_map<std::string,
            std::shared_ptr<const fem::Function<double>>>
            solutions_;
    };

} // namespace hellofem::app
