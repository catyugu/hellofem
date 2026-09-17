// hellofem::app — electrostatics field solver
// SPDX-License-Identifier: MIT

#include "electric.h"

#include "kernels.h"
#include "physics_field.h"

#include <spdlog/spdlog.h>

namespace hellofem::app {
    namespace {

        /// The electrostatics physics of a case: the model's conductivity with
        /// its voltage terminals and grounds, exporting the potential V.
        class ElectricField : public PhysicsField {
        public:
            ElectricField(const Physics& physics, CaseContext& ctx)
                : solver_(std::make_shared<ElectrostaticsSolver>(
                      ctx.mesh().mesh, ctx.mesh().facet_tags, ctx.mesh().cell_tags,
                      ctx.mesh().order))
                , variables_ {scalar_variable("V", "(V)", solver_->solution())}
            {
                // A material law of this physics may read its own dependent
                // variable: publish the solution before its coefficients.
                ctx.publish("V", solver_->solution());
                solver_->set_conductivity(
                    ctx.material_property("electricconductivity"));

                for (const PhysicsFeature& feature : physics.features) {
                    if (feature.type == "Terminal"
                        and feature.properties.contains("V0")) {
                        ScalarExpression voltage
                            = ctx.expression(feature.properties.at("V0"));
                        for (int id : feature.selection)
                            solver_->add_voltage_bc(id, voltage);
                    }
                    else if (feature.type == "Ground")
                        for (int id : feature.selection)
                            solver_->add_voltage_bc(id, ScalarExpression(0.0));
                }
                spdlog::info("electric: bound conductive media");
            }

            void initialize(double t0) override
            {
                solver_->constrain_solution(t0);
            }

            void solve_level(double t) override
            {
                solver_->solve_steady(t);
                spdlog::info("electric: V solved at t = {} s", t);
            }

            std::span<const Variable> variables() const override
            {
                return variables_;
            }

        private:
            std::shared_ptr<ElectrostaticsSolver> solver_;
            std::vector<Variable> variables_;
        };

        const FieldRegistration electric_field {"ConductiveMedia",
            [](const Physics& physics, CaseContext& ctx)
                -> std::unique_ptr<PhysicsField> {
                return std::make_unique<ElectricField>(physics, ctx);
            }};

    } // namespace

    // ---------------------------------------------------------------------------
    // ElectrostaticsSolver
    // ---------------------------------------------------------------------------

    ElectrostaticsSolver::ElectrostaticsSolver(
        std::shared_ptr<const mesh::Mesh<double>> mesh,
        std::shared_ptr<const mesh::MeshTags<int>> facet_tags,
        std::shared_ptr<const mesh::MeshTags<int>> cell_tags, int order)
        : FieldSolver(std::move(mesh), std::move(facet_tags), std::move(cell_tags),
              order, 1)
    {
    }

    void ElectrostaticsSolver::refresh(double t)
    {
        t_ = t;
        if (sigma_)
            sigma_->update(t);
    }

    void ElectrostaticsSolver::assemble_steady(la::MatrixCSR<double>& A,
        la::Vector<double>& b) const
    {
        auto bcs = make_bcs(voltages_, t_);
        auto rows = marked_rows(bcs);
        auto a = add_operator(A, rows, {sigma_->function()},
            kernels::diffusion_scalar);
        b.set(0.0);
        impose_dirichlet(A, b, a, bcs, rows);
    }

    void ElectrostaticsSolver::constrain_solution(double t)
    {
        impose_values(std::span(u_->x()->array()), make_bcs(voltages_, t));
    }

} // namespace hellofem::app
