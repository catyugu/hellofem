// hellofem::app — the model data a physics binds itself against
// SPDX-License-Identifier: MIT

#include "case_context.h"

#include "defaults.h"

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

    int CaseContext::element_order(
        const Physics& physics, std::string_view variable) const
    {
        const auto it = physics.element_order.find(std::string(variable));
        return it == physics.element_order.end() ? default_element_order
                                                 : it->second;
    }

    CaseContext::CaseContext(const ModelScript& model, const LoadedMesh& mesh,
        TimeSettings time)
        : model_(model)
        , mesh_(mesh)
        , time_(std::move(time))
    {
        for (const auto& p : model_.parameters)
            params_[p.name] = p.si;
    }

    ScalarExpression CaseContext::expression(std::string_view text) const
    {
        return ScalarExpression(text, params_);
    }

    std::shared_ptr<DomainProperty> CaseContext::zero_property() const
    {
        auto coefficient = std::make_shared<DomainProperty>(mesh_.mesh,
            mesh_.cell_tags, params_);
        // A material law may read any dependent variable of the model.
        for (const auto& [symbol, solution] : solutions_)
            coefficient->bind_field(symbol, solution);
        return coefficient;
    }

    std::shared_ptr<DomainProperty> CaseContext::property(
        const std::function<std::string(const Material&)>& value) const
    {
        auto coefficient = zero_property();
        for (int dom : domain_ids(*mesh_.cell_tags))
            if (const Material* material = model_.material_on_domain(dom)) {
                const std::string expression = value(*material);
                if (not expression.empty())
                    coefficient->set_expression(dom, expression);
            }
        return coefficient;
    }

    std::shared_ptr<DomainProperty> CaseContext::material_property(
        std::string_view name) const
    {
        return property([name](const Material& material) {
            const MaterialProperty* p = material.property(name);
            return p ? p->scalar_value() : std::string {};
        });
    }

    std::shared_ptr<DomainProperty> CaseContext::uniform_property(
        std::string_view text) const
    {
        auto coefficient = zero_property();
        for (int dom : domain_ids(*mesh_.cell_tags))
            coefficient->set_expression(dom, text);
        return coefficient;
    }

    std::shared_ptr<FacetProperty> CaseContext::facet_property(
        std::string_view text, int boundary) const
    {
        auto coefficient = std::make_shared<FacetProperty>(mesh_.mesh,
            mesh_.facet_tags, boundary, params_);
        // A boundary law may read any dependent variable of the model.
        for (const auto& [symbol, solution] : solutions_)
            coefficient->bind_field(symbol, solution);
        coefficient->set_expression(text);
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
