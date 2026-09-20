// hellofem::app — solid mechanics field solver
// SPDX-License-Identifier: MIT
#pragma once

#include "field.h"

#include <optional>
#include <set>

namespace hellofem::app {

    /// Solid mechanics: -div(C : eps(u)) = f_th (thermal expansion load).
    class SolidMechanicsSolver : public FieldSolver {
    public:
        SolidMechanicsSolver(std::shared_ptr<const mesh::Mesh<double>> mesh,
            std::shared_ptr<const mesh::MeshTags<int>> facet_tags,
            std::shared_ptr<const mesh::MeshTags<int>> cell_tags, int order);

        void set_elastic(std::shared_ptr<DomainProperty> E,
            std::shared_ptr<DomainProperty> nu)
        {
            E_ = std::move(E);
            nu_ = std::move(nu);
        }

        /// Thermal expansion load: sigma_th = C : (alpha (T - T_ref) I).
        void set_thermal_expansion(
            std::shared_ptr<const fem::Function<double>> T,
            std::shared_ptr<DomainProperty> alpha, double t_ref);

        /// Zero displacement on a boundary.
        void add_fixed_bc(int boundary_id) { fixed_.insert(boundary_id); }

        void refresh(double t) override;
        void assemble_steady(la::MatrixCSR<double>& A,
            la::Vector<double>& b) const override;
        void constrain_solution(double t) override;

    private:
        /// Zero-displacement conditions of the fixed boundaries, expanded to
        /// every component of their dofs.
        std::vector<fem::DirichletBC<double>> fixed_bcs() const;

        std::shared_ptr<DomainProperty> E_, nu_;
        struct ThermalExpansion {
            std::shared_ptr<const fem::Function<double>> T;
            std::shared_ptr<DomainProperty> alpha;
            double t_ref = 0.0;
        };
        std::optional<ThermalExpansion> thermal_expansion_;
        std::set<int> fixed_;
    };

} // namespace hellofem::app
