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
    /// every field prepares the first level, then the study's fields are
    /// advanced over its span by one step controller, whose step size and
    /// order the local error estimates of every field of the level select,
    /// and the result is stored at the times the store mode asks for — and
    /// exports it in COMSOL's Data format.
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

        /// Advance the study's fields over the whole span of `times` with one
        /// step controller: its step size and order follow from the local
        /// truncation error of every field of the level at once (see
        /// `BdfController`), its steps are placed as the step mode asks for,
        /// and the result is stored as the store mode asks for.
        ///
        /// The stepping starts from the state the fields prepared, advanced by
        /// the reference's consistent-initialization step — one artificial
        /// backward-Euler step of a fraction of the initial step, which also
        /// gives the time derivative the initial-step rule reads.
        void advance_adaptive(
            std::span<TimeStepper* const> steppers, const std::vector<double>& times);

        /// The step-error estimate of the study's fields at once: the
        /// reference's weighted root mean square over every dependent variable
        /// (see the definition), which is what the step is judged by.
        StepError combine_errors(std::span<TimeStepper* const> steppers) const;

        /// Store the output times of `times` that the step from `from` to `t`
        /// reached, as the store mode asks for: interpolated from the step
        /// that stepped over them, taken from the solver step closest to
        /// them, or the solver's own steps.
        /// @return the index of the next output time to store.
        std::size_t store_outputs(std::size_t output, double from, double t,
            const std::vector<double>& times);

        // --- result export ---
        struct Snapshot {
            double time = 0.0;
            std::vector<double> columns; // expression-major, per vertex
        };
        std::vector<double> evaluate_columns() const;
        void record(double t);

        /// Record the result at an output time that falls between two levels:
        /// the stepping fields take the value their scheme's polynomial gives
        /// there, the algebraic ones are solved at that state, and the levels
        /// are left as they were.
        void record_at(double t);

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
