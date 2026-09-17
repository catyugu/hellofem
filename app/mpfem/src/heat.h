// hellofem::app — heat transfer field solver
// SPDX-License-Identifier: MIT
#pragma once

#include "defaults.h"
#include "field.h"

#include <map>
#include <vector>

namespace hellofem::app {

    /// Heat transfer: rho cp dT/dt - div(k grad T) = Q, with a Joule source
    /// and Robin convection on the boundary.
    class HeatTransferSolver : public TimeDependentField {
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
            const TimeLevel& level) const override;

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
        ScalarExpression initial_ {reference_temperature};
    };

} // namespace hellofem::app
