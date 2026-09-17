// hellofem::app — case-level solver scheduler
// SPDX-License-Identifier: MIT
#pragma once

#include "case_context.h"
#include "mesh_loader.h"
#include "model_script.h"
#include "physics_field.h"

#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace hellofem::app {

    /// Drives a parsed COMSOL model over a loaded mesh: builds one field per
    /// physics interface of the model (see `register_field`, which the fields
    /// of the app do from their own translation units), solves the model's
    /// study — each field once for a stationary one, and each field once per
    /// output time, with the scheme for the fields a transient study advances
    /// in time — and exports the result in COMSOL's Data format.
    ///
    /// The scheduler names no physics: the fields, their order (the order of
    /// the model's physics interfaces), what they export and how they step
    /// come from the field kinds that registered themselves.
    class CaseScheduler {
    public:
        /// @param[in] scheme Time stepping scheme of a transient study (see
        /// `time_schemes`).
        CaseScheduler(const ModelScript& model, const LoadedMesh& mesh,
            std::string_view scheme);

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

        // --- result export ---
        struct Snapshot {
            double time = 0.0;
            std::vector<double> columns; // expression-major, per vertex
        };
        std::vector<double> evaluate_columns() const;
        void record(double t);

        void run_stationary();
        void run_transient();

        // --- data ---
        ModelScript model_;
        LoadedMesh lm_;
        CaseContext ctx_;

        std::shared_ptr<const mesh::Mesh<double>> mesh_;
        std::vector<std::unique_ptr<PhysicsField>> fields_;
        std::vector<Snapshot> snapshots_;
    };

} // namespace hellofem::app
