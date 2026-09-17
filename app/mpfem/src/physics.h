// hellofem::app — physics field solvers (electrostatics / heat / solid)
// SPDX-License-Identifier: MIT
#pragma once

#include "property.h"
#include "time_scheme.h"

#include "fem/DirichletBC.h"
#include "fem/Form.h"
#include "fem/Function.h"
#include "fem/FunctionSpace.h"
#include "la/MatrixCSR.h"
#include "la/SparsityPattern.h"
#include "la/Vector.h"
#include "mesh/Mesh.h"
#include "mesh/MeshTags.h"

#include <map>
#include <memory>
#include <set>
#include <vector>

namespace hellofem::app {

    /// One time level of a multistep scheme: the weights that apply at it,
    /// the level time, and the stored data of the previous levels.
    struct TimeLevel {
        TimeWeights weights;
        /// Time of this level (its Dirichlet data is evaluated there).
        double time = 0.0;
        /// Previous solution levels, most recent first.
        std::span<const la::Vector<double>* const> history;
        /// Source load of the previous level (null at the first step).
        const la::Vector<double>* source_old = nullptr;
        /// Source load of this level; stored for the next step.
        la::Vector<double>* source_new = nullptr;
    };

    /// Base for a single-physics field solver: owns the function space, the
    /// solution, the sparsity pattern of the linearized system and the mesh
    /// topology queries the physics needs.
    class FieldSolver {
    public:
        FieldSolver(std::shared_ptr<const mesh::Mesh<double>> mesh,
            std::shared_ptr<const mesh::MeshTags<int>> facet_tags,
            std::shared_ptr<const mesh::MeshTags<int>> cell_tags, int order,
            int value_dim);
        virtual ~FieldSolver() = default;

        /// Evaluate the model data that depends on time or on the solution
        /// (material properties, boundary values, source terms) at time `t`,
        /// taking the current solution as the field state. Call before
        /// assembling a system.
        virtual void refresh(double t) = 0;

        /// Whether a material property reads the solution, i.e. whether the
        /// linearized system has to be iterated to convergence.
        virtual bool nonlinear() const = 0;

        /// Assemble the linearized steady system `A u = b` at the current
        /// state, with the Dirichlet conditions imposed.
        virtual void assemble_steady(la::MatrixCSR<double>& A,
            la::Vector<double>& b) const = 0;

        /// Impose the Dirichlet data of time `t` on the current solution. A
        /// state that is not the result of a solve (the initial one of a
        /// transient run) must satisfy the pointwise constraints.
        virtual void constrain_solution(double t) = 0;

        std::shared_ptr<fem::Function<double>> solution() const { return u_; }

        std::shared_ptr<fem::FunctionSpace<double>> space() const { return V_; }

        /// Sparsity pattern of the linearized system.
        const la::SparsityPattern& pattern() const { return *pattern_; }

        /// All cells, the assembly range.
        std::vector<std::int32_t> cells() const;

    protected:
        /// Dofs on the facets carrying the given 1-based boundary ids.
        std::vector<std::int32_t> boundary_dofs(const std::set<int>& ids) const;

        /// `(cell, local facet)` pairs of the given 1-based boundary ids.
        std::vector<std::int32_t> boundary_facets(const std::set<int>& ids) const;

        /// Dirichlet conditions of `(boundary id -> value)` at time `t`.
        std::vector<fem::DirichletBC<double>> make_bcs(
            const std::map<int, ScalarExpression>& values, double t) const;

        std::shared_ptr<const mesh::Mesh<double>> mesh_;
        std::shared_ptr<const mesh::MeshTags<int>> facet_tags_;
        std::shared_ptr<const mesh::MeshTags<int>> cell_tags_;
        std::shared_ptr<fem::FunctionSpace<double>> V_;
        std::shared_ptr<fem::Function<double>> u_;
        std::shared_ptr<la::SparsityPattern> pattern_;
        int order_;
        int value_dim_;
        double t_ = 0.0; // time of the last refresh
    };

    /// Electrostatics: -div(sigma grad V) = 0.
    /// BCs: voltage (Dirichlet), electric insulation (natural).
    class ElectrostaticsSolver : public FieldSolver {
    public:
        ElectrostaticsSolver(std::shared_ptr<const mesh::Mesh<double>> mesh,
            std::shared_ptr<const mesh::MeshTags<int>> facet_tags,
            std::shared_ptr<const mesh::MeshTags<int>> cell_tags, int order);

        void set_conductivity(std::shared_ptr<CellProperty> sigma)
        {
            sigma_ = std::move(sigma);
        }

        /// V = value(t) on a boundary.
        void add_voltage_bc(int boundary_id, ScalarExpression value)
        {
            voltages_[boundary_id] = std::move(value);
        }

        void refresh(double t) override;
        bool nonlinear() const override
        {
            return sigma_ and sigma_->field_dependent();
        }
        void assemble_steady(la::MatrixCSR<double>& A,
            la::Vector<double>& b) const override;
        void constrain_solution(double t) override;

    private:
        std::shared_ptr<CellProperty> sigma_;
        std::map<int, ScalarExpression> voltages_;
    };

    /// Heat transfer: rho cp dT/dt - div(k grad T) = Q, with a Joule source
    /// and Robin convection on the boundary.
    class HeatTransferSolver : public FieldSolver {
    public:
        HeatTransferSolver(std::shared_ptr<const mesh::Mesh<double>> mesh,
            std::shared_ptr<const mesh::MeshTags<int>> facet_tags,
            std::shared_ptr<const mesh::MeshTags<int>> cell_tags, int order);

        void set_conductivity(std::shared_ptr<CellProperty> k)
        {
            k_ = std::move(k);
        }
        void set_thermal_mass(std::shared_ptr<CellProperty> rho_cp)
        {
            rho_cp_ = std::move(rho_cp);
        }
        void set_source(std::shared_ptr<CellProperty> Q) { Q_ = std::move(Q); }

        /// Joule heating source from an electric solution: adds
        /// ∫ sigma |grad V|² phi to the heat load.
        void set_joule_source(std::shared_ptr<const fem::Function<double>> V,
            std::shared_ptr<CellProperty> sigma);

        /// T = value(t) on a boundary.
        void add_temperature_bc(int boundary_id, ScalarExpression value)
        {
            temps_[boundary_id] = std::move(value);
        }

        /// Robin condition h(t) (T - Tinf(t)) on a boundary. Both
        /// coefficients are per-cell properties (uniform boundary data is
        /// one expression on every cell).
        void add_convection(int boundary_id, std::shared_ptr<CellProperty> h,
            std::shared_ptr<CellProperty> t_inf)
        {
            convections_.push_back(
                {boundary_id, std::move(h), std::move(t_inf)});
        }

        /// Initial temperature of a transient run (the model's initial-value
        /// expression). Without one, COMSOL's heat-transfer default of
        /// 293.15 K applies.
        void set_initial_temperature(ScalarExpression value)
        {
            initial_ = std::move(value);
        }

        /// Set the solution to the initial temperature at t = 0.
        void apply_initial_condition();

        void refresh(double t) override;
        bool nonlinear() const override;
        void assemble_steady(la::MatrixCSR<double>& A,
            la::Vector<double>& b) const override;
        void constrain_solution(double t) override;

        /// Assemble one time step of the heat equation with the scheme
        /// weights `w`:
        ///   A = a0 M + b0 K,
        ///   b = c_new f_new + c_old f_old - Σ_{k>=1} (a_k M + b_k K) u_k,
        /// where M is the thermal-mass operator and K the conductivity plus
        /// convection operator, both at the current (refreshed) state. The
        /// Dirichlet data is taken at the level time `t`.
        void assemble_step(la::MatrixCSR<double>& A, la::Vector<double>& b,
            const TimeLevel& level) const;

    private:
        struct Convection {
            int boundary_id;
            std::shared_ptr<CellProperty> h;
            std::shared_ptr<CellProperty> t_inf;
        };

        /// Assembly pieces shared by the steady and the transient path.
        void assemble_sources(la::Vector<double>& f) const;

        std::shared_ptr<CellProperty> k_, rho_cp_, Q_;
        std::shared_ptr<const fem::Function<double>> joule_V_;
        std::shared_ptr<CellProperty> joule_sigma_;
        std::map<int, ScalarExpression> temps_;
        std::vector<Convection> convections_;
        ScalarExpression initial_ {293.15};
    };

    /// Solid mechanics: -div(C : eps(u)) = f_th (thermal expansion load).
    class SolidMechanicsSolver : public FieldSolver {
    public:
        SolidMechanicsSolver(std::shared_ptr<const mesh::Mesh<double>> mesh,
            std::shared_ptr<const mesh::MeshTags<int>> facet_tags,
            std::shared_ptr<const mesh::MeshTags<int>> cell_tags, int order);

        void set_elastic(std::shared_ptr<CellProperty> E,
            std::shared_ptr<CellProperty> nu)
        {
            E_ = std::move(E);
            nu_ = std::move(nu);
        }

        /// Thermal expansion load: sigma_th = C : (alpha (T - T_ref) I).
        void set_thermal_expansion(
            std::shared_ptr<const fem::Function<double>> T,
            std::shared_ptr<CellProperty> alpha, double t_ref);

        /// Zero displacement on a boundary.
        void add_fixed_bc(int boundary_id) { fixed_.insert(boundary_id); }

        void refresh(double t) override;
        bool nonlinear() const override;
        void assemble_steady(la::MatrixCSR<double>& A,
            la::Vector<double>& b) const override;
        void constrain_solution(double t) override;

    private:
        /// Zero-displacement conditions of the fixed boundaries, expanded to
        /// every component of their dofs.
        std::vector<fem::DirichletBC<double>> fixed_bcs() const;

        std::shared_ptr<CellProperty> E_, nu_;
        struct Thermal {
            std::shared_ptr<const fem::Function<double>> T;
            std::shared_ptr<CellProperty> alpha;
            double t_ref = 0.0;
        };
        std::optional<Thermal> thermal_;
        std::set<int> fixed_;
    };

} // namespace hellofem::app
