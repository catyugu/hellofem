// hellofem::app — the model data a physics binds itself against
// SPDX-License-Identifier: MIT
#pragma once

#include "field.h"
#include "mesh_loader.h"
#include "model_script.h"

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
    class CaseContext {
    public:
        /// @param[in] scheme Time stepping scheme of a transient study (see
        /// `time_schemes`).
        CaseContext(const ModelScript& model, const LoadedMesh& mesh,
            std::string_view scheme);

        const ModelScript& model() const { return model_; }
        const LoadedMesh& mesh() const { return mesh_; }
        std::string_view scheme() const { return scheme_; }

        /// Compile a model expression with the parameters bound by value.
        ScalarExpression expression(std::string_view text) const;

        /// A cell coefficient: `value(material)` gives its expression on the
        /// domains of that material (an empty string leaves a domain without
        /// one, i.e. at the zero of the coefficient).
        std::shared_ptr<CellProperty> property(
            const std::function<std::string(const Material&)>& value) const;

        /// The material property `name`, on every domain whose material
        /// defines one.
        std::shared_ptr<CellProperty> material_property(
            std::string_view name) const;

        /// One expression on every cell. A boundary coefficient is one
        /// expression on all cells: the facet kernels read the coefficient of
        /// the adjacent cell.
        std::shared_ptr<CellProperty> uniform_property(
            std::string_view text) const;

        /// A cell coefficient without a model value: zero on every domain,
        /// for a source a physics fills per domain from its own features.
        std::shared_ptr<CellProperty> zero_property() const;

        /// Publish the solution of a field under `symbol`, the COMSOL
        /// dependent-variable name every model expression refers to it by.
        void publish(std::string_view symbol,
            std::shared_ptr<const fem::Function<double>> solution);

        /// The solution published under `symbol`, or nullptr.
        std::shared_ptr<const fem::Function<double>> solution(
            std::string_view symbol) const;

    private:
        /// Bind the solutions published so far to a coefficient.
        void bind_solutions(CellProperty& coefficient) const;

        const ModelScript& model_;
        const LoadedMesh& mesh_;
        std::string scheme_;
        std::unordered_map<std::string, double> params_;
        std::unordered_map<std::string,
            std::shared_ptr<const fem::Function<double>>>
            solutions_;
    };

} // namespace hellofem::app
