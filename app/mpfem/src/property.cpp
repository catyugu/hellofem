// hellofem::app — model material properties and scalar model expressions
// SPDX-License-Identifier: MIT

#include "property.h"

#include "basis/element-families.h"
#include "fem/dofmapbuilder.h"
#include "mesh/cell_types.h"
#include "mesh/utils.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <numeric>
#include <stdexcept>

namespace hellofem::app {
    namespace {

        using B = basis::element::family;
        using LV = basis::element::lagrange_variant;
        using DV = basis::element::dpc_variant;

        /// Names the expression parser already provides (coordinates, time).
        bool reserved_name(const std::string& name)
        {
            return name == "x" or name == "y" or name == "z" or name == "t";
        }

    } // namespace

    // ---------------------------------------------------------------------------
    // ScalarExpression
    // ---------------------------------------------------------------------------

    ScalarExpression::ScalarExpression(std::string_view text,
        const std::unordered_map<std::string, double>& params)
    {
        auto impl = std::make_shared<Impl>();
        impl->vars = params;
        std::unordered_map<std::string, double*> vars;
        for (auto& [name, value] : impl->vars)
            if (not reserved_name(name))
                vars[name] = &value;
        impl->expr.parse(text, vars);
        impl_ = std::move(impl);
    }

    // ---------------------------------------------------------------------------
    // CellProperty
    // ---------------------------------------------------------------------------

    CellProperty::CellProperty(std::shared_ptr<const mesh::Mesh<double>> mesh,
        std::shared_ptr<const mesh::MeshTags<int>> cell_tags,
        std::unordered_map<std::string, double> params)
        : mesh_(std::move(mesh))
        , cell_tags_(std::move(cell_tags))
        , vars_(std::move(params))
    {
        for (auto& [name, value] : vars_)
            if (not reserved_name(name))
                var_ptrs_[name] = &value;

        // DG0 space: one scalar dof per cell, from the element's own dof
        // layout (the coordinate element's layout is P1).
        auto cell_type = mesh_->topology()->cell_type();
        auto ctype = mesh::cell_type_to_basix_type(cell_type);
        auto fe = std::make_shared<fem::FiniteElement<double>>(
            basis::create_element<double>(B::P, ctype, 0, LV::equispaced,
                DV::unset, true));
        auto layout = fe->create_dof_layout();
        auto [imap, bs, dofmaps] = fem::build_dofmap_data(*mesh_->topology(),
            {layout}, nullptr);
        auto dmap = std::make_shared<fem::DofMap>(layout,
            std::make_shared<common::IndexMap>(std::move(imap)), bs,
            std::move(dofmaps.front()), bs);
        auto V = std::make_shared<fem::FunctionSpace<double>>(mesh_, fe, dmap);
        f_ = std::make_shared<fem::Function<double>>(V);
        f_->x()->set(0.0);
    }

    std::int32_t CellProperty::num_cells() const
    {
        return f_->function_space()->dofmap()->map().extent(0);
    }

    /// 1-based domain of a cell (1 when the mesh tags are absent, i.e. the
    /// whole mesh is one domain).
    int CellProperty::domain_of_cell(std::int32_t cell) const
    {
        if (not cell_tags_)
            return 1;
        const auto& indices = cell_tags_->indices();
        auto it = std::lower_bound(indices.begin(), indices.end(), cell);
        if (it == indices.end() or *it != cell)
            return 0;
        return cell_tags_->values()[static_cast<std::size_t>(
            std::distance(indices.begin(), it))];
    }

    double& CellProperty::cell_entry(std::int32_t cell)
    {
        return f_->x()->array()[static_cast<std::size_t>(
            f_->function_space()->dofmap()->cell_dofs(cell).front())];
    }

    void CellProperty::set_value(int dom, double value)
    {
        DomainValue& dv = values_[dom];
        dv.value = value;
        dv.expr.reset();
        const std::int32_t nc = num_cells();
        for (std::int32_t c = 0; c < nc; ++c)
            if (domain_of_cell(c) == dom)
                cell_entry(c) = value;
    }

    void CellProperty::set_expression(int dom, std::string_view text)
    {
        auto expr = std::make_shared<Expression>();
        expr->parse(text, var_ptrs_);
        for (const std::string& used : expr->variables())
            for (const auto& field : fields_)
                if (field.symbol == used) {
                    field_dependent_ = true;
                    spdlog::debug("property on domain {}: '{}' reads the field '{}'",
                        dom, text, used);
                }
        DomainValue& dv = values_[dom];
        dv.expr = std::move(expr);
        dynamic_ = true;
    }

    void CellProperty::bind_field(std::string symbol,
        std::shared_ptr<const fem::Function<double>> field)
    {
        // The field takes over the variable slot: a bound field shadows a
        // model parameter of the same name.
        vars_[symbol] = 0.0;
        var_ptrs_[symbol] = &vars_[symbol];
        fields_.push_back(BoundField {std::move(symbol), std::move(field), {}});
    }

    void CellProperty::cell_centroids()
    {
        if (centroids_ready_)
            return;
        const auto& topo = *mesh_->topology();
        auto c_to_v = topo.connectivity(topo.dim(), 0);
        const auto [vc, vshape] = mesh::compute_vertex_coords(*mesh_);
        const std::size_t nv = vshape[1];
        const std::int32_t nc = num_cells();
        centroids_.assign(static_cast<std::size_t>(3 * nc), 0.0);
        for (std::int32_t c = 0; c < nc; ++c) {
            auto verts = c_to_v->links(c);
            for (auto v : verts)
                for (int d = 0; d < 3; ++d)
                    centroids_[static_cast<std::size_t>(3 * c + d)]
                        += vc[d * nv + v];
            for (int d = 0; d < 3; ++d)
                centroids_[static_cast<std::size_t>(3 * c + d)]
                    /= static_cast<double>(verts.size());
        }
        centroids_ready_ = true;
    }

    void CellProperty::update(double t)
    {
        if (not dynamic_)
            return;
        cell_centroids();
        const std::int32_t nc = num_cells();

        if (not fields_.empty()) {
            const std::array<std::size_t, 2> shape {
                static_cast<std::size_t>(nc), 3};
            for (auto& field : fields_) {
                auto [vals, vshape] = field.function->eval(centroids_, shape);
                if (vshape[1] != 1)
                    throw std::runtime_error(
                        "CellProperty: a bound field must be scalar");
                field.values = std::move(vals);
            }
        }

        for (std::int32_t c = 0; c < nc; ++c) {
            auto it = values_.find(domain_of_cell(c));
            if (it == values_.end() or not it->second.expr)
                continue;
            for (auto& field : fields_)
                *var_ptrs_[field.symbol]
                    = field.values[static_cast<std::size_t>(c)];
            cell_entry(c) = it->second.expr->eval(centroids_[3 * c],
                centroids_[3 * c + 1], centroids_[3 * c + 2], t);
        }
    }

} // namespace hellofem::app
