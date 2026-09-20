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
#include <stdexcept>
#include <unordered_set>

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

        /// A DG0 function on `mesh`: one scalar value per cell, which is what
        /// a cell or a facet integral reads its coefficient as.
        std::shared_ptr<fem::Function<double>> make_dg0_function(
            const std::shared_ptr<const mesh::Mesh<double>>& mesh)
        {
            // DG0 space: one scalar dof per cell, from the element's own dof
            // layout (the coordinate element's layout is P1).
            auto cell_type = mesh->topology()->cell_type();
            auto ctype = mesh::cell_type_to_basix_type(cell_type);
            auto fe = std::make_shared<fem::FiniteElement<double>>(
                basis::create_element<double>(B::P, ctype, 0, LV::equispaced,
                    DV::unset, true));
            auto layout = fe->create_dof_layout();
            auto [imap, bs, dofmaps] = fem::build_dofmap_data(*mesh->topology(),
                {layout}, nullptr);
            auto dmap = std::make_shared<fem::DofMap>(layout,
                std::make_shared<common::IndexMap>(std::move(imap)), bs,
                std::move(dofmaps.front()), bs);
            auto V = std::make_shared<fem::FunctionSpace<double>>(mesh, fe, dmap);
            auto f = std::make_shared<fem::Function<double>>(V);
            f->x()->set(0.0);
            return f;
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
    // DomainProperty
    // ---------------------------------------------------------------------------

    DomainProperty::DomainProperty(
        std::shared_ptr<const mesh::Mesh<double>> mesh,
        std::shared_ptr<const mesh::MeshTags<int>> cell_tags,
        std::unordered_map<std::string, double> params)
        : mesh_(std::move(mesh))
        , cell_tags_(std::move(cell_tags))
        , vars_(std::move(params))
    {
        bind_parameters(vars_, var_ptrs_);
        f_ = make_dg0_function(mesh_);
    }

    std::int32_t DomainProperty::num_cells() const
    {
        return f_->function_space()->dofmap()->map().extent(0);
    }

    /// The expression of the domain of every cell: the update loop reads it
    /// per cell, and a cell whose domain defines no expression keeps a null.
    void DomainProperty::resolve_cell_expressions()
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

    double& DomainProperty::cell_entry(std::int32_t cell)
    {
        return f_->x()->array()[static_cast<std::size_t>(
            f_->function_space()->dofmap()->cell_dofs(cell).front())];
    }

    void DomainProperty::set_expression(int dom, std::string_view text)
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

    void DomainProperty::bind_field(std::string symbol,
        std::shared_ptr<const fem::Function<double>> field)
    {
        // The field takes over the variable slot: a bound field shadows a
        // model parameter of the same name.
        vars_[symbol] = 0.0;
        var_ptrs_[symbol] = &vars_[symbol];
        fields_.push_back(BoundField {std::move(symbol), std::move(field), {}});
    }

    void DomainProperty::cell_centroids()
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

    void DomainProperty::update(double t)
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
                    "DomainProperty: a bound field must be scalar");
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

    // ---------------------------------------------------------------------------
    // FacetProperty
    // ---------------------------------------------------------------------------

    FacetProperty::FacetProperty(
        std::shared_ptr<const mesh::Mesh<double>> mesh,
        std::shared_ptr<const mesh::MeshTags<int>> facet_tags, int boundary,
        std::unordered_map<std::string, double> params)
        : mesh_(std::move(mesh))
        , facet_tags_(std::move(facet_tags))
        , boundary_(boundary)
        , vars_(std::move(params))
    {
        bind_parameters(vars_, var_ptrs_);
        f_ = make_dg0_function(mesh_);
    }

    double& FacetProperty::cell_entry(std::int32_t cell)
    {
        return f_->x()->array()[static_cast<std::size_t>(
            f_->function_space()->dofmap()->cell_dofs(cell).front())];
    }

    /// The point the value is read at on every facet of the boundary, and the
    /// cells that carry it: the one next to the facet, and the one across it
    /// when the facet is interior.
    void FacetProperty::resolve_sites()
    {
        if (not facet_tags_)
            return;
        const int tdim = mesh_->topology()->dim();
        auto f_to_c = mesh_->topology()->connectivity(tdim - 1, tdim);
        auto f_to_v = mesh_->topology()->connectivity(tdim - 1, 0);
        const auto [vc, vshape] = mesh::compute_vertex_coords(*mesh_);
        const std::size_t nv = vshape[1];

        const auto& indices = facet_tags_->indices();
        const auto& boundaries = facet_tags_->values();
        std::unordered_set<std::int32_t> reached;
        for (std::size_t i = 0; i < indices.size(); ++i) {
            if (boundaries[i] != boundary_)
                continue;
            const std::int32_t facet = indices[i];
            auto vertices = f_to_v->links(facet);
            std::array<double, 3> point {0.0, 0.0, 0.0};
            for (std::int32_t v : vertices)
                for (int d = 0; d < 3; ++d)
                    point[static_cast<std::size_t>(d)]
                        += vc[static_cast<std::size_t>(d) * nv
                            + static_cast<std::size_t>(v)];
            for (double& c : point)
                c /= static_cast<double>(vertices.size());
            site_points_.insert(site_points_.end(), point.begin(), point.end());

            auto cells = f_to_c->links(facet);
            site_cells_.push_back(cells[0]);
            site_neighbours_.push_back(
                cells.size() > 1 ? cells[1] : std::int32_t {-1});

            // A value that varies over the boundary is one value per facet,
            // and the DG0 coefficient a facet integral reads carries one value
            // per cell: a cell the boundary reaches through two facets cannot
            // carry both.
            if (varies_over_boundary_)
                for (std::int32_t cell : cells)
                    if (not reached.insert(cell).second)
                        throw std::runtime_error("FacetProperty: boundary "
                            + std::to_string(boundary_) + " reaches cell "
                            + std::to_string(cell)
                            + " through two facets, and its value varies over "
                              "the boundary");
        }
    }

    void FacetProperty::set_expression(std::string_view text)
    {
        auto expr = std::make_shared<Expression>();
        expr->parse(text, var_ptrs_);
        varies_over_boundary_ = false;
        for (const std::string& used : expr->variables()) {
            bool bound = false;
            for (auto& field : fields_)
                if (field.symbol == used) {
                    field.used = true;
                    field_dependent_ = true;
                    bound = true;
                }
            // The coordinates and a bound field vary over the boundary; a
            // model parameter and the time do not.
            varies_over_boundary_ = varies_over_boundary_ or bound
                or (reserved_variable(used) and used != "t");
        }
        expression_ = std::move(expr);
        site_points_.clear();
        site_cells_.clear();
        site_neighbours_.clear();
    }

    void FacetProperty::bind_field(std::string symbol,
        std::shared_ptr<const fem::Function<double>> field)
    {
        // The field takes over the variable slot: a bound field shadows a
        // model parameter of the same name.
        vars_[symbol] = 0.0;
        var_ptrs_[symbol] = &vars_[symbol];
        fields_.push_back(BoundField {std::move(symbol), std::move(field), {}});
    }

    void FacetProperty::update(double t)
    {
        if (not expression_)
            return;
        if (site_cells_.empty())
            resolve_sites();
        if (site_cells_.empty())
            throw std::runtime_error("FacetProperty: boundary "
                + std::to_string(boundary_) + " has no facet on the mesh");
        const std::size_t nsites = site_cells_.size();

        for (auto& field : fields_) {
            if (not field.used)
                continue;
            // A law reads its field through one variable, so only a scalar
            // solution has a value to bind.
            if (field.function->function_space()->element()->value_size() != 1)
                throw std::runtime_error(
                    "FacetProperty: a bound field must be scalar");
            field.values.assign(nsites, 0.0);
            field.function->eval(site_points_, {nsites, 3}, site_cells_,
                field.values, {nsites, 1});
        }

        for (std::size_t i = 0; i < nsites; ++i) {
            for (auto& field : fields_)
                if (field.used)
                    *var_ptrs_[field.symbol] = field.values[i];
            const double value = expression_->eval(site_points_[3 * i],
                site_points_[3 * i + 1], site_points_[3 * i + 2], t);
            cell_entry(site_cells_[i]) = value;
            if (site_neighbours_[i] >= 0)
                cell_entry(site_neighbours_[i]) = value;
        }
    }

} // namespace hellofem::app
