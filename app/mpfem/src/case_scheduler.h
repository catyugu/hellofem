// hellofem::app — case-level solver scheduler
// SPDX-License-Identifier: MIT
#pragma once

#include "mesh_loader.h"
#include "model_script.h"
#include "physics.h"

#include <memory>
#include <string>

namespace hellofem::app {

    /// High-level dispatcher that reads a model script + mesh, selects
    /// solver strategies for each physics field, drives the solve sequence
    /// (including nonlinear iterations and multiphysics coupling), and
    /// exports results in COMSOL-compatible format.
    ///
    /// Usage:
    ///   CaseScheduler sched(model_script, loaded_mesh);
    ///   sched.run();
    ///   sched.export_result("result.txt");
    class CaseScheduler {
    public:
        CaseScheduler(const ModelScript& model, const LoadedMesh& mesh);

        /// Run the full solve sequence:
        ///   1. Electric currents -> V (if ConductiveMedia present)
        ///   2. Heat transfer   -> T (if HeatTransfer present; Picard if alpha_k)
        ///   3. Solid mechanics -> u (if SolidMechanics present)
        void run();

        /// The solved fields (nullptr if not solved).
        std::shared_ptr<fem::Function<double>> V() const { return V_; }
        std::shared_ptr<fem::Function<double>> T() const { return T_; }
        std::shared_ptr<fem::Function<double>> u() const { return u_; }

        /// Write a COMSOL-style Data export file.
        void export_result(const std::string& path) const;

        /// Access the loaded mesh.
        std::shared_ptr<const mesh::Mesh<double>> mesh() const { return mesh_; }

    private:
        // --- Per-field solvers ---
        void solve_electric();
        void solve_heat();
        void solve_solid();

        // --- Nonlinear heat dispatch ---
        void solve_heat_picard(double k0, double ak, double tref,
            std::shared_ptr<CellProperty> cp_k);

        // --- Helpers ---
        void fill_heat_bcs();
        std::shared_ptr<CellProperty> make_ht_property(const std::string& prop);

        // --- Data ---
        ModelScript model_;
        LoadedMesh lm_;

        std::shared_ptr<const mesh::Mesh<double>> mesh_;
        std::shared_ptr<const mesh::MeshTags<int>> facet_tags_;
        std::shared_ptr<const mesh::MeshTags<int>> cell_tags_;
        int order_;

        std::unordered_map<std::string, double> params_;

        // Solver instances
        std::shared_ptr<ElectrostaticsSolver> es_;
        std::shared_ptr<HeatTransferSolver> ht_;
        std::shared_ptr<SolidMechanicsSolver> sm_;

        // Solved fields
        std::shared_ptr<fem::Function<double>> V_;
        std::shared_ptr<fem::Function<double>> T_;
        std::shared_ptr<fem::Function<double>> u_;

        // Heat-transfer BC/coupling state
        bool ht_joule_ = false;
        std::shared_ptr<CellProperty> ht_sigma_;
    };

} // namespace hellofem::app
