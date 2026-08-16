// hellofem::app — physics field solvers (electrostatics / heat / solid)
// SPDX-License-Identifier: MIT
#pragma once

#include "fem/Function.h"
#include "fem/FunctionSpace.h"
#include "la/MatrixCSR.h"
#include "mesh/Mesh.h"
#include "mesh/MeshTags.h"

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace hellofem::app {

    /// A scalar material property as a cell-wise Function (DG0) over the
    /// mesh. The user assigns a value per domain; the function holds a
    /// per-cell value read by the assembly kernels.
    class CellProperty {
    public:
        CellProperty(std::shared_ptr<const mesh::Mesh<double>> mesh,
            std::shared_ptr<const mesh::MeshTags<int>> cell_tags);

        /// Set the value on domain `dom` (1-based COMSOL id).
        void set_domain(int dom, double value);

        /// Recompute from an evaluator callable `(x,y,z,t) -> double` for
        /// every cell centroid on the given domains (or all if empty).
        void fill_from(std::function<double(double, double, double, double)> eval,
            const std::set<int>& domains, double t);

        std::shared_ptr<fem::Function<double>> function() const { return f_; }

    private:
        std::shared_ptr<fem::Function<double>> f_;
        std::shared_ptr<const mesh::Mesh<double>> mesh_;
        std::shared_ptr<const mesh::MeshTags<int>> cell_tags_;
    };

    /// Base for a single-physics field solver. Owns the function space,
    /// the solution function and the per-cell material properties.
    class FieldSolver {
    public:
        FieldSolver(std::shared_ptr<const mesh::Mesh<double>> mesh,
            std::shared_ptr<const mesh::MeshTags<int>> facet_tags,
            std::shared_ptr<const mesh::MeshTags<int>> cell_tags,
            int order, int value_dim);
        virtual ~FieldSolver() = default;

        /// Solve the steady problem and write into `u_`.
        virtual void solve_steady() = 0;

        /// Current solution coefficients.
        std::shared_ptr<fem::Function<double>> solution() const { return u_; }

        std::shared_ptr<fem::FunctionSpace<double>> space() const { return V_; }

    protected:
        std::shared_ptr<const mesh::Mesh<double>> mesh_;
        std::shared_ptr<const mesh::MeshTags<int>> facet_tags_;
        std::shared_ptr<const mesh::MeshTags<int>> cell_tags_;
        std::shared_ptr<fem::FunctionSpace<double>> V_;
        std::shared_ptr<fem::Function<double>> u_;
        int order_;
        int value_dim_;
    };

    /// Electrostatics: -div(sigma grad V) = 0.
    /// BCs: voltage (Dirichlet), electric insulation (natural).
    class ElectrostaticsSolver : public FieldSolver {
    public:
        ElectrostaticsSolver(std::shared_ptr<const mesh::Mesh<double>> mesh,
            std::shared_ptr<const mesh::MeshTags<int>> facet_tags,
            std::shared_ptr<const mesh::MeshTags<int>> cell_tags, int order);

        void set_conductivity(std::shared_ptr<CellProperty> sigma) { sigma_ = sigma; }
        void add_voltage_bc(int boundary_id, double value) { voltages_[boundary_id] = value; }

        void solve_steady() override;

        /// Joule heating source Q = sigma |grad V|^2 (cell property for
        /// assembly of the heat RHS).
        std::shared_ptr<CellProperty> joule_heat() { return joule_; }

    private:
        std::shared_ptr<CellProperty> sigma_;
        std::map<int, double> voltages_;
        std::shared_ptr<CellProperty> joule_;
    };

    /// Heat transfer: rho cp dT/dt - div(k grad T) = Q.
    /// Steady: -div(k grad T) = Q with Robin convection on the boundary.
    class HeatTransferSolver : public FieldSolver {
    public:
        HeatTransferSolver(std::shared_ptr<const mesh::Mesh<double>> mesh,
            std::shared_ptr<const mesh::MeshTags<int>> facet_tags,
            std::shared_ptr<const mesh::MeshTags<int>> cell_tags, int order);

        void set_conductivity(std::shared_ptr<CellProperty> k) { k_ = k; }
        void set_thermal_mass(std::shared_ptr<CellProperty> rho_cp) { rho_cp_ = rho_cp; }
        void set_source(std::shared_ptr<CellProperty> Q) { Q_ = Q; }
        /// Robin BC: h (T - Tinf) on boundary id.
        void add_convection(int boundary_id, double h, double Tinf);
        void add_temperature_bc(int boundary_id, double value) { temps_[boundary_id] = value; }

        void solve_steady() override;

        /// Solve one backward-Euler step: (M + dt K) T^{n+1} = M T^n + dt F.
        void solve_step(double dt);

    private:
        std::shared_ptr<CellProperty> k_, rho_cp_, Q_;
        std::map<int, double> temps_;
        struct Convection { int id; double h; double Tinf; };
        std::vector<Convection> convections_;
    };

    /// Solid mechanics: -div(C : eps(u)) = f_th, thermal strain RHS.
    class SolidMechanicsSolver : public FieldSolver {
    public:
        SolidMechanicsSolver(std::shared_ptr<const mesh::Mesh<double>> mesh,
            std::shared_ptr<const mesh::MeshTags<int>> facet_tags,
            std::shared_ptr<const mesh::MeshTags<int>> cell_tags, int order);

        void set_elastic(std::shared_ptr<CellProperty> E, std::shared_ptr<CellProperty> nu) { E_ = E; nu_ = nu; }
        /// Set the thermal strain load: sigma_th (Voigt, 6 comps) per cell.
        void set_thermal_strain(std::shared_ptr<CellProperty> sigma_th) { sigma_th_ = sigma_th; }
        void add_fixed_bc(int boundary_id) { fixed_.insert(boundary_id); }

        void solve_steady() override;

    private:
        std::shared_ptr<CellProperty> E_, nu_, sigma_th_;
        std::set<int> fixed_;
    };

} // namespace hellofem::app
