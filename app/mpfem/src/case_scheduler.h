// hellofem::app — case-level solver scheduler
// SPDX-License-Identifier: MIT
#pragma once

#include "case_context.h"
#include "mesh_loader.h"
#include "model_script.h"
#include "physics_field.h"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace hellofem::app {

    /// Drives a parsed COMSOL model over a loaded mesh: builds one field per
    /// physics interface of the model (see `FieldRegistration`, which each
    /// physics holds in its own translation unit), solves the model's study —
    /// every field prepares the first level, then solves one level per
    /// output time, which is a step of its time scheme for the fields a
    /// transient study advances, at a step size and order the local error
    /// estimate selects — and exports the result
    /// in COMSOL's Data format.
    ///
    /// The scheduler names no physics: the fields, their order (the order of
    /// the model's physics interfaces), what they export and what steps in
    /// time come from the field kinds that registered themselves.
    class CaseScheduler {
    public:
        /// @param[in] time Time stepping of a transient study: its scheme
        /// and the tolerance its steps are held to (see `TimeSettings`).
        CaseScheduler(const ModelScript& model, const LoadedMesh& mesh,
            TimeSettings time);

        /// Run the study.
        void run();

        /// Write a COMSOL-style Data export: one column per model
        /// expression, and one column group per stored time level for a
        /// transient study.
        void export_result(const std::string& path) const;

    private:
        /// The variable the model's export expression `name` refers to, or
        /// nullptr when no field exports it.
        const Variable* variable(std::string_view name) const;

        /// The unit of the variable `name`, as COMSOL writes it.
        std::string unit(std::string_view name) const;

        /// The cell every mesh vertex is read on (a vertex is a dof of each
        /// cell around it, so any one of them reads the same value), with
        /// the vertex coordinates in the point-major layout `Variable::eval`
        /// takes. Both are fixed for the whole run.
        void resolve_vertices();

        /// Advance the steppers over the whole span of `times` with the step
        /// size and the order the local truncation error estimate selects
        /// (see `step_factor` and `next_order`), recording the result at each
        /// output time. The steps are held to the output times, so those are
        /// solved rather than interpolated: only a step that is too coarse is
        /// rejected, and the last one of an interval is taken at whatever
        /// size is left.
        void advance_adaptive(
            std::span<TimeStepper* const> steppers, const std::vector<double>& times);

        // --- result export ---
        struct Snapshot {
            double time = 0.0;
            std::vector<double> columns; // expression-major, per vertex
        };
        std::vector<double> evaluate_columns() const;
        void record(double t);

        // --- data ---
        ModelScript model_;
        LoadedMesh lm_;
        TimeSettings time_;
        CaseContext ctx_;

        std::shared_ptr<const mesh::Mesh<double>> mesh_;
        std::vector<std::unique_ptr<PhysicsField>> fields_;
        std::vector<Snapshot> snapshots_;

        // Export points: the vertex coordinates, and the cell each of them
        // is read on.
        std::vector<double> vertex_points_;
        std::vector<std::int32_t> vertex_cells_;
    };

} // namespace hellofem::app
