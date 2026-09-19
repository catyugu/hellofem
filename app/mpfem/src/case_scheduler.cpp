// hellofem::app — case-level solver scheduler implementation
// SPDX-License-Identifier: MIT

#include "case_scheduler.h"

#include "mesh/utils.h"
#include "spdlog/spdlog.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <span>
#include <stdexcept>
#include <utility>

namespace hellofem::app {

    CaseScheduler::CaseScheduler(const ModelScript& model, const LoadedMesh& lm,
        TimeSettings time)
        : model_(model)
        , lm_(lm)
        , time_(std::move(time))
        , ctx_(model_, lm_, time_)
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
            spdlog::info("transient: {} output times, steps held to a local error "
                         "of {} relative",
                times.size(), time_.tolerance);

        // The first level is the state the fields prepare for themselves: the
        // initial values of a field the study advances, and the constraints
        // of the fields it does not. A field that is algebraic — one that
        // does not step in time — is solved there like at any other level.
        const double t0 = times.empty() ? 0.0 : times.front();
        for (const auto& field : fields_)
            field->initialize(t0);
        for (const auto& field : fields_)
            if (not field->stepper())
                field->solve_level(t0);
        record(t0);

        std::vector<TimeStepper*> steppers;
        for (const auto& field : fields_)
            if (TimeStepper* stepper = field->stepper())
                steppers.push_back(stepper);

        // A transient study is stepped by the error test of its scheme: the
        // output times are the times its steps are held to, not its steps.
        if (model_.study.transient)
            advance_adaptive(steppers, times);
    }

    void CaseScheduler::advance_adaptive(
        std::span<TimeStepper* const> steppers, const std::vector<double>& times)
    {
        const double t_end = times.back();
        const double span = t_end - times.front();
        BdfController control(time_, span);

        // The first level of the stepping: the state the fields prepared,
        // advanced by one artificial backward-Euler step of a fraction of the
        // initial step. That is the reference's consistent initialization — it
        // reconciles the initial values with the constraints of the level, and
        // its increment is the time derivative the initial-step rule reads.
        double t = times.front();
        const double probe
            = span * first_step_fraction * backward_euler_step_fraction;
        for (TimeStepper* stepper : steppers)
            stepper->step(t + probe, 1);
        double derivative_norm = 0.0;
        for (TimeStepper* stepper : steppers)
            derivative_norm = std::max(derivative_norm, stepper->derivative_norm());
        t += probe;
        control.start(control.first_step(derivative_norm));
        spdlog::info("transient: first step {} s (derivative norm {}), "
                     "largest step {} s, orders {}..{}",
            control.step_size(), derivative_norm, span * max_step_fraction,
            time_.min_order, time_.max_order);

        // The state at times.front() is already stored; an output time the
        // initialization step itself reached is not (it is not a step of the
        // stepping, and the mode counts those).
        std::size_t output = store_outputs(1, times.front(), t, times);
        bool landed = false; // a step has landed inside the current subinterval
        int steps = 0;
        int rejected = 0;

        while (t < t_end) {
            const double next = output < times.size() ? times[output] : t_end;
            double h = control.step_size();
            if (time_.steps == StepsMode::strict) {
                // Every step ends at an output time; the solver takes the
                // steps its tolerance needs in between.
                h = std::min(h, next - t);
            }
            else if (time_.steps == StepsMode::intermediate and not landed
                and t + h >= next) {
                // Every subinterval of the output times holds at least one
                // step: this one is cut short to land inside the current one.
                h = 0.5 * (next - t);
            }
            if (not time_.interpolate_end_time)
                h = std::min(h, t_end - t);

            const double from = t;
            for (TimeStepper* stepper : steppers)
                stepper->step(t + h, control.order());

            StepError error;
            if (control.error_controlled()) {
                error = combine_errors(steppers);
                if (error.at > 1.0) {
                    // The step missed the tolerance: the fields go back to the
                    // level it started from, and it is retried smaller.
                    for (TimeStepper* stepper : steppers)
                        stepper->undo();
                    control.reject(error, h);
                    ++rejected;
                    if (control.step_size() <= control.smallest_step())
                        throw std::runtime_error(
                            "CaseScheduler: the time step at t = "
                            + std::to_string(t) + " s fell to "
                            + std::to_string(control.step_size())
                            + " s and the local error is still "
                            + std::to_string(error.at) + "; the tolerance "
                            + std::to_string(time_.tolerance)
                            + " cannot be met on this interval");
                    continue;
                }
            }

            t += h;
            // A field that does not step is algebraic: it follows every
            // level, not only the stored times.
            for (const auto& field : fields_)
                if (not field->stepper())
                    field->solve_level(t);
            control.accept(error, h);
            ++steps;

            if (time_.store == StoreMode::steps)
                record(t);
            output = store_outputs(output, from, t, times);
            landed = output < times.size() and t < times[output];
        }

        spdlog::info("transient: {} steps over {} s ({} rejected)", steps, span,
            rejected);
    }

    StepError CaseScheduler::combine_errors(
        std::span<TimeStepper* const> steppers) const
    {
        // The reference's norm is the weighted root mean square over every
        // dependent variable of the study, `(1/M) sum_j (1/N_j) sum_i
        // (e_i/W_i)^2`, so a field enters it as its own mean square divided by
        // the number of the study's fields — not as its norm taken against the
        // others'. A field the study does not advance contributes nothing: the
        // app solves it exactly at each level, so it carries no truncation
        // error of its own (the reference's such fields do carry a small one,
        // measured at a thousandth of the temperature's).
        //
        // Taking the largest field's norm instead is not the same thing and
        // costs steps: measured on EcTSmBusbarTransient, it holds the step at
        // 9.6 s where the reference doubles, and lands 4 of the 63 columns
        // outside the case's tolerance where this norm lands all of them
        // inside it, on the reference's own sequence of steps.
        const double fields = static_cast<double>(fields_.size());
        StepError combined;
        for (TimeStepper* stepper : steppers) {
            const StepError field = stepper->error_estimates();
            combined.below += field.below * field.below / fields;
            combined.at += field.at * field.at / fields;
            combined.above += field.above * field.above / fields;
        }
        combined.below = std::sqrt(combined.below);
        combined.at = std::sqrt(combined.at);
        combined.above = std::sqrt(combined.above);
        return combined;
    }

    std::size_t CaseScheduler::store_outputs(std::size_t output, double from,
        double t, const std::vector<double>& times)
    {
        while (output < times.size() and times[output] <= t) {
            const double target = times[output];
            if (time_.store == StoreMode::interpolate)
                record_at(target);
            else if (time_.store == StoreMode::closest) {
                // The solver step closest to the output time, among the steps
                // that passed it: the one just taken, or the level it started
                // from.
                record_at(std::abs(target - from) < std::abs(target - t) ? from
                                                                         : target);
            }
            // With the solver's own steps stored, the output times are only
            // counted here — the modes that place the steps need them, and
            // their own solutions are the steps themselves.
            ++output;
        }
        return output;
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

    void CaseScheduler::record_at(double t)
    {
        for (const auto& field : fields_)
            if (TimeStepper* stepper = field->stepper())
                stepper->interpolate(t, stepper->solution());
        for (const auto& field : fields_)
            if (not field->stepper())
                field->solve_level(t);
        record(t);
        for (const auto& field : fields_)
            if (TimeStepper* stepper = field->stepper())
                stepper->restore(stepper->solution());
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
        out << "% Length unit:        " << model_.length_unit << "\n";
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
            // The coordinates go out in the model's geometry length unit,
            // which is what COMSOL's own data export writes.
            const double s = model_.length_scale();
            out << vertex_points_[3 * i] / s << "  "
                << vertex_points_[3 * i + 1] / s << "  "
                << vertex_points_[3 * i + 2] / s;
            for (const Snapshot& snapshot : snapshots_)
                for (std::size_t k = 0; k < nk; ++k)
                    out << "  " << snapshot.columns[i * nk + k];
            out << "\n";
        }
    }

} // namespace hellofem::app
