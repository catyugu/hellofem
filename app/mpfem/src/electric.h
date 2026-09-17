// hellofem::app — electrostatics field solver
// SPDX-License-Identifier: MIT
#pragma once

#include "field.h"

#include <map>

namespace hellofem::app {

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

} // namespace hellofem::app
