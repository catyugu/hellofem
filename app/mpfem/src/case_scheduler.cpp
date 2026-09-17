// hellofem::app — case-level solver scheduler implementation
// SPDX-License-Identifier: MIT

#include "case_scheduler.h"

#include "mesh/utils.h"
#include "spdlog/spdlog.h"

#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace hellofem::app {

    CaseScheduler::CaseScheduler(const ModelScript& model, const LoadedMesh& lm,
        std::string_view scheme)
        : model_(model)
        , lm_(lm)
        , ctx_(model_, lm_, scheme)
        , mesh_(lm.mesh)
    {
        // One field per physics interface of the model, in the model's own
        // order: that order is the solve order of a level.
        for (const Physics& physics : model_.physics) {
            const FieldKind* kind = field_kind(physics.type);
            if (not kind)
                throw std::runtime_error(
                    "CaseScheduler: the physics '" + physics.type
                    + "' has no registered field");
            fields_.push_back(kind->create(physics, ctx_));
        }

        // Couplings pair two physics of the model, so they are bound once
        // every field has published its solution.
        for (const auto& field : fields_)
            field->bind_couplings(ctx_);

        resolve_vertices();
    }

    void CaseScheduler::resolve_vertices()
    {
        auto [vc, vshape] = mesh::compute_vertex_coords(*mesh_);
        const std::size_t nv = vshape[1];

        // The vertex coordinates are component-major; the evaluator wants
        // point-major rows.
        vertex_points_.resize(nv * 3);
        for (std::size_t i = 0; i < nv; ++i)
            for (int q = 0; q < 3; ++q)
                vertex_points_[i * 3 + static_cast<std::size_t>(q)]
                    = vc[q * nv + i];

        const int tdim = mesh_->topology()->dim();
        auto topo = mesh_->topology_mutable();
        topo->create_entities(0);
        topo->create_connectivity(tdim, 0);
        auto c_to_v = topo->connectivity(tdim, 0);
        vertex_cells_.assign(nv, -1);
        for (std::int32_t c = 0;
            c < static_cast<std::int32_t>(c_to_v->num_nodes()); ++c)
            for (auto v : c_to_v->links(c))
                if (vertex_cells_[static_cast<std::size_t>(v)] < 0)
                    vertex_cells_[static_cast<std::size_t>(v)] = c;
    }

    void CaseScheduler::run()
    {
        const std::vector<double>& times = model_.study.times;
        if (model_.study.transient and times.size() < 2)
            throw std::runtime_error(
                "CaseScheduler: a transient study needs at least two output times");
        if (model_.study.transient)
            spdlog::info("transient: {} output times", times.size());

        // The first level is the state the fields prepare for themselves: the
        // initial values of a field the study advances, and the constraints
        // of the fields it does not. The fields it does not advance solve
        // that level here, exactly as they solve every later one; a transient
        // study then steps to each following output time.
        const double t0 = times.empty() ? 0.0 : times.front();
        for (const auto& field : fields_)
            field->initialize(t0);
        for (const auto& field : fields_)
            if (not field->advances_in_time())
                field->solve_level(t0);
        record(t0);

        for (std::size_t i = 1; i < times.size(); ++i) {
            for (const auto& field : fields_)
                field->solve_level(times[i]);
            record(times[i]);
        }
    }

    // -------------------------------------------------------------------------
    // Result export
    // -------------------------------------------------------------------------

    const Variable* CaseScheduler::variable(std::string_view name) const
    {
        for (const auto& field : fields_)
            for (const Variable& variable : field->variables())
                if (variable.name == name)
                    return &variable;
        return nullptr;
    }

    std::string CaseScheduler::unit(std::string_view name) const
    {
        const Variable* found = variable(name);
        return found ? found->unit : "(1)";
    }

    std::vector<double> CaseScheduler::evaluate_columns() const
    {
        const std::size_t nv = vertex_cells_.size();
        const std::size_t nk = model_.export_config.expressions.size();
        std::vector<double> columns(nv * nk, 0.0);

        std::vector<double> values(nv);
        for (std::size_t k = 0; k < nk; ++k) {
            const std::string& name = model_.export_config.expressions[k];
            const Variable* found = variable(name);
            if (not found) {
                spdlog::warn("export '{}' not supported; writes 0", name);
                continue;
            }
            found->eval(vertex_points_, vertex_cells_, values);
            for (std::size_t i = 0; i < nv; ++i)
                columns[i * nk + k] = values[i];
        }
        return columns;
    }

    void CaseScheduler::record(double t)
    {
        Snapshot snapshot;
        snapshot.time = t;
        snapshot.columns = evaluate_columns();
        snapshots_.push_back(std::move(snapshot));
    }

    void CaseScheduler::export_result(const std::string& path) const
    {
        const std::size_t nv = vertex_cells_.size();
        const std::size_t nk = model_.export_config.expressions.size();
        const std::size_t nt = snapshots_.size();
        if (nt == 0)
            throw std::runtime_error("CaseScheduler: no result to export");
        const bool per_time = model_.study.transient;

        std::ofstream out(path);
        if (!out)
            throw std::runtime_error("cannot open " + path);
        out << std::setprecision(17);
        out << "% Model:              " << model_.name << ".mph\n";
        out << "% Version:            COMSOL 6.2.0.290\n";
        out << "% Dimension:          3\n";
        out << "% Nodes:              " << nv << "\n";
        out << "% Expressions:        " << nk * nt << "\n";
        out << "% Description:        ";
        for (std::size_t k = 0; k < nk; ++k)
            out << (k ? ", " : "") << model_.export_config.expressions[k];
        out << "\n% Length unit:        m\n";
        out << "% x                       y                        z                        ";
        for (const Snapshot& snapshot : snapshots_)
            for (std::size_t k = 0; k < nk; ++k) {
                out << model_.export_config.expressions[k] << " "
                    << unit(model_.export_config.expressions[k]);
                if (per_time)
                    out << " @ t=" << snapshot.time;
                out << "  ";
            }
        out << "\n";
        for (std::size_t i = 0; i < nv; ++i) {
            out << vertex_points_[3 * i] << "  " << vertex_points_[3 * i + 1]
                << "  " << vertex_points_[3 * i + 2];
            for (const Snapshot& snapshot : snapshots_)
                for (std::size_t k = 0; k < nk; ++k)
                    out << "  " << snapshot.columns[i * nk + k];
            out << "\n";
        }
    }

} // namespace hellofem::app
