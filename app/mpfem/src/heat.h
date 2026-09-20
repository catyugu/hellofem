// hellofem::app — heat transfer field solver
// SPDX-License-Identifier: MIT
#pragma once

#include "defaults.h"
#include "field.h"

#include <map>
#include <set>
#include <vector>

namespace hellofem::app {

    /// Heat transfer: rho cp dT/dt - div(k grad T) = Q, with a Joule source
    /// and Robin convection on the boundary.
    class HeatTransferSolver : public TimeDependentField {
    public:
        HeatTransferSolver(std::shared_ptr<const mesh::Mesh<double>> mesh,
            std::shared_ptr<const mesh::MeshTags<int>> facet_tags,
            std::shared_ptr<const mesh::MeshTags<int>> cell_tags, int order);

        void set_conductivity(std::shared_ptr<DomainProperty> k)
        {
            k_ = std::move(k);
        }
        void set_thermal_mass(std::shared_ptr<DomainProperty> rho_cp)
        {
            rho_cp_ = std::move(rho_cp);
        }

        /// A volumetric heat source `Q` on the given 1-based domain ids: the
        /// source belongs to the domains its feature selects, and a domain no
        /// feature selects carries none.
        void add_source(const std::set<int>& domains,
            std::shared_ptr<DomainProperty> Q);

        /// Joule heating source from an electric solution: adds
        /// ∫ sigma |grad V|² phi to the heat load.
        void set_joule_source(std::shared_ptr<const fem::Function<double>> V,
            std::shared_ptr<DomainProperty> sigma);

        /// T = value(t) on a boundary.
        void add_temperature_bc(int boundary_id, ScalarExpression value)
        {
            temps_[boundary_id] = std::move(value);
        }

        /// Robin condition h (T - Tinf) on the boundary both coefficients
        /// belong to.
        void add_convection(std::shared_ptr<FacetProperty> h,
            std::shared_ptr<FacetProperty> t_inf);

        /// A thin layer on the boundary of `ds`: a shell of thickness `ds`
        /// and conductivity `k`, thermally thin — the tangential conduction
        /// `ds k grad_t T . grad_t phi` joins the operator and no temperature
        /// difference builds up across the layer's thickness. A boundary
        /// inside the domain (an imprinted face) carries the layer too.
        void add_thin_layer(std::shared_ptr<FacetProperty> ds,
            std::shared_ptr<FacetProperty> k);

        /// Initial temperature of a transient run (the model's initial-value
        /// expression). Without one, COMSOL's default (see
        /// `reference_temperature`) applies.
        void set_initial_temperature(ScalarExpression value)
        {
            initial_ = std::move(value);
        }

        /// Set the solution to the initial temperature at the study's start
        /// time `t0`.
        void apply_initial_condition(double t0);

        void refresh(double t) override;
        void assemble_steady(la::MatrixCSR<double>& A,
            la::Vector<double>& b, double t) const override;
        void constrain_solution(double t) override;

        /// Assemble one time step of the heat equation with the scheme
        /// weights `w`:
        ///   A = a0 M + b0 K,
        ///   b = f - Σ_{k>=1} (a_k M + b_k K) u_k
        void assemble_step(la::MatrixCSR<double>& A, la::Vector<double>& b,
            const TimeLevel& level) const override;

    private:
        struct Convection {
            std::shared_ptr<FacetProperty> h;
            std::shared_ptr<FacetProperty> t_inf;
        };

        struct ThinLayer {
            std::shared_ptr<FacetProperty> ds;
            std::shared_ptr<FacetProperty> k;
        };

        /// A volumetric heat source over the domains it belongs to.
        struct Source {
            std::set<int> domains;
            std::shared_ptr<DomainProperty> Q;
        };

        /// Assembly pieces shared by the steady and the transient path.
        void assemble_sources(la::Vector<double>& f) const;

        std::shared_ptr<DomainProperty> k_, rho_cp_;
        std::shared_ptr<const fem::Function<double>> joule_V_;
        std::shared_ptr<DomainProperty> joule_sigma_;
        std::vector<Source> sources_;
        std::map<int, ScalarExpression> temps_;
        std::vector<Convection> convections_;
        std::vector<ThinLayer> thin_layers_;
        ScalarExpression initial_ {reference_temperature};
    };

} // namespace hellofem::app
