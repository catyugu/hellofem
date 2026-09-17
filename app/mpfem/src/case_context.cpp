// hellofem::app — the model data a physics binds itself against
// SPDX-License-Identifier: MIT

#include "case_context.h"

#include <set>

namespace hellofem::app {
    namespace {

        /// 1-based domain ids of the tagged cells.
        std::set<int> domain_ids(const mesh::MeshTags<int>& cell_tags)
        {
            std::set<int> ids;
            for (int value : cell_tags.values())
                ids.insert(value);
            return ids;
        }

    } // namespace

    CaseContext::CaseContext(const ModelScript& model, const LoadedMesh& mesh,
        std::string_view scheme)
        : model_(model)
        , mesh_(mesh)
        , scheme_(scheme)
    {
        for (const auto& p : model_.parameters)
            params_[p.name] = p.si;
    }

    ScalarExpression CaseContext::expression(std::string_view text) const
    {
        return ScalarExpression(text, params_);
    }

    void CaseContext::bind_solutions(CellProperty& coefficient) const
    {
        for (const auto& [symbol, solution] : solutions_)
            coefficient.bind_field(symbol, solution);
    }

    std::shared_ptr<CellProperty> CaseContext::property(
        const std::function<std::string(const Material&)>& value) const
    {
        auto coefficient = std::make_shared<CellProperty>(mesh_.mesh,
            mesh_.cell_tags, params_);
        bind_solutions(*coefficient);
        for (int dom : domain_ids(*mesh_.cell_tags))
            if (const Material* material = model_.material_on_domain(dom)) {
                const std::string expression = value(*material);
                if (not expression.empty())
                    coefficient->set_expression(dom, expression);
            }
        return coefficient;
    }

    std::shared_ptr<CellProperty> CaseContext::material_property(
        std::string_view name) const
    {
        return property([name](const Material& material) {
            const MaterialProperty* p = material.property(name);
            return p ? p->scalar_value() : std::string {};
        });
    }

    std::shared_ptr<CellProperty> CaseContext::uniform_property(
        std::string_view text) const
    {
        auto coefficient = std::make_shared<CellProperty>(mesh_.mesh,
            mesh_.cell_tags, params_);
        bind_solutions(*coefficient);
        for (int dom : domain_ids(*mesh_.cell_tags))
            coefficient->set_expression(dom, text);
        return coefficient;
    }

    std::shared_ptr<CellProperty> CaseContext::zero_property() const
    {
        auto coefficient = std::make_shared<CellProperty>(mesh_.mesh,
            mesh_.cell_tags, params_);
        bind_solutions(*coefficient);
        return coefficient;
    }

    void CaseContext::publish(std::string_view symbol,
        std::shared_ptr<const fem::Function<double>> solution)
    {
        solutions_[std::string(symbol)] = std::move(solution);
    }

    std::shared_ptr<const fem::Function<double>> CaseContext::solution(
        std::string_view symbol) const
    {
        const auto it = solutions_.find(std::string(symbol));
        return it == solutions_.end() ? nullptr : it->second;
    }

} // namespace hellofem::app
