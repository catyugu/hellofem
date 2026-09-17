// hellofem::app — model material properties and scalar model expressions
// SPDX-License-Identifier: MIT

#include "property.h"

#include "basis/element-families.h"
#include "fem/dofmapbuilder.h"
#include "mesh/cell_types.h"
#include "mesh/utils.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <stdexcept>

namespace hellofem::app {
    namespace {

        using B = basis::element::family;
        using LV = basis::element::lagrange_variant;
        using DV = basis::element::dpc_variant;

        /// Bind the variable storage the parsed expressions of a coefficient
        /// read: one pointer per model parameter, keyed by name. The names the
        /// parser provides itself stay unbound, so its own coordinates win.
        void bind_parameters(std::unordered_map<std::string, double>& storage,
            std::unordered_map<std::string, double*>& bound)
        {
            for (auto& [name, value] : storage)
                if (not reserved_variable(name))
                    bound[name] = &value;
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
        bind_parameters(impl->vars, vars);
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
        bind_parameters(vars_, var_ptrs_);

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

    /// The expression of the domain of every cell: the update loop reads it
    /// per cell, and a cell whose domain defines no expression keeps a null.
    void CellProperty::resolve_cell_expressions()
    {
        cell_expressions_.assign(static_cast<std::size_t>(num_cells()), nullptr);
        if (not cell_tags_) {
            // The whole mesh is one domain (COMSOL domain 1).
            const auto it = values_.find(1);
            if (it != values_.end())
                std::ranges::fill(cell_expressions_, it->second.get());
            return;
        }
        const auto& indices = cell_tags_->indices();
        const auto& domains = cell_tags_->values();
        for (std::size_t i = 0; i < indices.size(); ++i) {
            const auto it = values_.find(domains[i]);
            if (it != values_.end())
                cell_expressions_[static_cast<std::size_t>(indices[i])]
                    = it->second.get();
        }
    }

    double& CellProperty::cell_entry(std::int32_t cell)
    {
        return f_->x()->array()[static_cast<std::size_t>(
            f_->function_space()->dofmap()->cell_dofs(cell).front())];
    }

    void CellProperty::set_expression(int dom, std::string_view text)
    {
        auto expr = std::make_shared<Expression>();
        expr->parse(text, var_ptrs_);
        for (const std::string& used : expr->variables())
            for (auto& field : fields_)
                if (field.symbol == used) {
                    field.used = true;
                    field_dependent_ = true;
                    spdlog::debug("property on domain {}: '{}' reads the field '{}'",
                        dom, text, used);
                }
        values_[dom] = std::move(expr);
        cell_expressions_.clear();
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
        centroid_cells_.resize(static_cast<std::size_t>(nc));
        for (std::int32_t c = 0; c < nc; ++c) {
            auto verts = c_to_v->links(c);
            for (auto v : verts)
                for (int d = 0; d < 3; ++d)
                    centroids_[static_cast<std::size_t>(3 * c + d)]
                        += vc[d * nv + v];
            for (int d = 0; d < 3; ++d)
                centroids_[static_cast<std::size_t>(3 * c + d)]
                    /= static_cast<double>(verts.size());
            centroid_cells_[static_cast<std::size_t>(c)] = c;
        }
        centroids_ready_ = true;
    }

    void CellProperty::update(double t)
    {
        if (values_.empty())
            return;
        cell_centroids();
        if (cell_expressions_.empty())
            resolve_cell_expressions();
        const std::int32_t nc = num_cells();

        for (auto& field : fields_) {
            if (not field.used)
                continue;
            // A law reads its field through one variable, so only a scalar
            // solution has a value to bind.
            if (field.function->function_space()->element()->value_size() != 1)
                throw std::runtime_error(
                    "CellProperty: a bound field must be scalar");
            field.values.assign(static_cast<std::size_t>(nc), 0.0);
            field.function->eval(centroids_, {static_cast<std::size_t>(nc), 3},
                centroid_cells_, field.values,
                {static_cast<std::size_t>(nc), 1});
        }

        for (std::int32_t c = 0; c < nc; ++c) {
            Expression* expression = cell_expressions_[static_cast<std::size_t>(c)];
            if (not expression)
                continue;
            for (auto& field : fields_)
                if (field.used)
                    *var_ptrs_[field.symbol]
                        = field.values[static_cast<std::size_t>(c)];
            cell_entry(c) = expression->eval(centroids_[3 * c],
                centroids_[3 * c + 1], centroids_[3 * c + 2], t);
        }
    }

} // namespace hellofem::app
