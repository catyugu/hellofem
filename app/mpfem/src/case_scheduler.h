// hellofem::app — case-level solver scheduler
// SPDX-License-Identifier: MIT
#pragma once

#include "mesh_loader.h"
#include "model_script.h"
#include "physics.h"
#include "time_scheme.h"
#include "transient.h"

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace hellofem::app {

    /// Drives a parsed COMSOL model over a loaded mesh: binds the physics of
    /// the model's study (materials, boundary conditions, multiphysics
    /// couplings), solves it — a stationary solve, or a time stepping loop
    /// with a time scheme for a transient study — and exports the result in
    /// COMSOL's Data format.
    class CaseScheduler {
    public:
        CaseScheduler(const ModelScript& model, const LoadedMesh& mesh);

        /// Time stepping scheme of a transient study. Default "bdf2".
        void set_time_scheme(std::string_view name);

        /// Run the study.
        void run();

        /// Final solution of each field (nullptr when the model has no such
        /// physics).
        std::shared_ptr<fem::Function<double>> V() const { return V_; }
        std::shared_ptr<fem::Function<double>> T() const { return T_; }
        std::shared_ptr<fem::Function<double>> u() const { return u_; }

        /// Write a COMSOL-style Data export: one column per model
        /// expression, and one column group per stored time level for a
        /// transient study.
        void export_result(const std::string& path) const;

        std::shared_ptr<const mesh::Mesh<double>> mesh() const { return mesh_; }

    private:
        // --- model binding (once, before the first solve) ---
        std::shared_ptr<CellProperty> make_property(const std::string& name);
        std::shared_ptr<CellProperty> make_thermal_mass();
        std::shared_ptr<CellProperty> make_boundary_property(std::string_view text);
        ScalarExpression expr(std::string_view text) const;

        void bind_electric();
        void bind_heat();
        void bind_solid();

        // --- one time level ---
        void solve_electric(double t);
        void solve_heat(double t);
        void solve_solid(double t);

        void run_stationary();
        void run_transient();

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

        std::shared_ptr<const mesh::Mesh<double>> mesh_;
        std::shared_ptr<const mesh::MeshTags<int>> facet_tags_;
        std::shared_ptr<const mesh::MeshTags<int>> cell_tags_;
        int order_;
        std::unordered_map<std::string, double> params_;

        std::shared_ptr<ElectrostaticsSolver> es_;
        std::shared_ptr<HeatTransferSolver> ht_;
        std::shared_ptr<SolidMechanicsSolver> sm_;
        std::shared_ptr<CellProperty> sigma_;

        std::shared_ptr<fem::Function<double>> V_;
        std::shared_ptr<fem::Function<double>> T_;
        std::shared_ptr<fem::Function<double>> u_;

        // Transient state.
        std::string scheme_name_ = "bdf2";
        std::unique_ptr<HeatTimeStepper> stepper_;
        std::vector<Snapshot> snapshots_;
    };

} // namespace hellofem::app
