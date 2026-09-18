// hellofem::app — solid mechanics field solver
// SPDX-License-Identifier: MIT

#include "solid.h"

#include "defaults.h"
#include "kernels.h"
#include "physics_field.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace hellofem::app {
    namespace {

        /// The solid mechanics physics of a case: the model's elastic
        /// properties with its fixed boundaries over the displacement,
        /// exporting its magnitude. It owns the thermal expansion coupling.
        class SolidField : public PhysicsField {
        public:
            SolidField(const Physics& physics, CaseContext& ctx)
                : solver_(std::make_shared<SolidMechanicsSolver>(
                      ctx.mesh().mesh, ctx.mesh().facet_tags, ctx.mesh().cell_tags,
                      ctx.element_order(physics, "displacement")))
                , variables_ {Variable {"solid.disp", "(m)",
                      [displacement = solver_->solution()](
                          std::span<const double> points,
                          std::span<const std::int32_t> cells,
                          std::span<double> values) {
                          std::vector<double> components(values.size() * 3, 0.0);
                          displacement->eval(points, {values.size(), 3}, cells,
                              components, {values.size(), 3});
                          for (std::size_t i = 0; i < values.size(); ++i)
                              values[i] = std::sqrt(
                                  components[3 * i] * components[3 * i]
                                  + components[3 * i + 1] * components[3 * i + 1]
                                  + components[3 * i + 2] * components[3 * i + 2]);
                      }}}
            {
                solver_->set_elastic(ctx.material_property("E"),
                    ctx.material_property("nu"));

                for (const PhysicsFeature& feature : physics.features)
                    if (feature.type == "Fixed")
                        for (int id : feature.selection)
                            solver_->add_fixed_bc(id);
                spdlog::info("solid: bound solid mechanics");
            }

            /// The thermal expansion coupling: sigma_th = C : (alpha (T -
            /// T_ref) I), with the temperature of the heat field of the case.
            void bind_couplings(CaseContext& ctx) override
            {
                for (const MultiphysicsCoupling& coupling : ctx.model().couplings) {
                    if (coupling.type != "ThermalExpansion")
                        continue;
                    auto temperature = ctx.solution("T");
                    if (not temperature)
                        throw std::runtime_error(
                            "ThermalExpansion without a temperature 'T'");
                    double t_ref = reference_temperature;
                    if (coupling.properties.contains(
                            "minput_strainreferencetemperature"))
                        t_ref = ctx
                                    .expression(coupling.properties.at(
                                        "minput_strainreferencetemperature"))
                                    .eval(0, 0, 0, 0);
                    solver_->set_thermal_expansion(temperature,
                        ctx.material_property("thermalexpansioncoefficient"),
                        t_ref);
                }
            }

            void initialize(double t0) override
            {
                solver_->constrain_solution(t0);
            }

            void solve_level(double t) override
            {
                solver_->solve_steady(t);
                spdlog::info("solid: u solved at t = {} s", t);
            }

            std::span<const Variable> variables() const override
            {
                return variables_;
            }

        private:
            std::shared_ptr<SolidMechanicsSolver> solver_;
            std::vector<Variable> variables_;
        };

        const FieldRegistration solid_field {"SolidMechanics",
            [](const Physics& physics, CaseContext& ctx)
                -> std::unique_ptr<PhysicsField> {
                return std::make_unique<SolidField>(physics, ctx);
            }};

    } // namespace

    // ---------------------------------------------------------------------------
    // SolidMechanicsSolver
    // ---------------------------------------------------------------------------

    SolidMechanicsSolver::SolidMechanicsSolver(
        std::shared_ptr<const mesh::Mesh<double>> mesh,
        std::shared_ptr<const mesh::MeshTags<int>> facet_tags,
        std::shared_ptr<const mesh::MeshTags<int>> cell_tags, int order)
        : FieldSolver(std::move(mesh), std::move(facet_tags), std::move(cell_tags),
              order, 3)
    {
        // Elasticity is the one operator of the app that a factorization suits
        // better than the AMG iteration: a three-component field on a
        // second-order mesh gives a large, ill-conditioned system, and the
        // blocked AMG hierarchy needs far more Krylov iterations for it than
        // for a scalar field. Measured on every solid case: see the report.
        linear_.solver_type = "direct";
    }

    void SolidMechanicsSolver::set_thermal_expansion(
        std::shared_ptr<const fem::Function<double>> T,
        std::shared_ptr<CellProperty> alpha, double t_ref)
    {
        thermal_ = Thermal {std::move(T), std::move(alpha), t_ref};
    }

    void SolidMechanicsSolver::refresh(double t)
    {
        t_ = t;
        if (E_)
            E_->update(t);
        if (nu_)
            nu_->update(t);
        if (thermal_)
            thermal_->alpha->update(t);
    }

    bool SolidMechanicsSolver::nonlinear() const
    {
        return solution_dependent(E_, nu_, thermal_ ? thermal_->alpha : nullptr);
    }

    void SolidMechanicsSolver::assemble_steady(la::MatrixCSR<double>& A,
        la::Vector<double>& b) const
    {
        auto bcs = fixed_bcs();
        auto rows = marked_rows(bcs);
        auto a = add_operator(A, rows, {E_->function(), nu_->function()},
            kernels::elasticity);

        // RHS: thermal expansion load ∫ Bᵀ sigma_th.
        b.set(0.0);
        if (thermal_) {
            auto t_ref = std::make_shared<fem::Constant<double>>(thermal_->t_ref);
            add_load(b,
                {thermal_->T, thermal_->alpha->function(), E_->function(),
                    nu_->function()},
                kernels::thermal_expansion_load, {std::move(t_ref)});
        }

        impose_dirichlet(A, b, a, bcs, rows);
    }

    std::vector<fem::DirichletBC<double>> SolidMechanicsSolver::fixed_bcs() const
    {
        // A fixed boundary fixes every component of its dofs.
        std::vector<std::int32_t> dofs = boundary_dofs(fixed_);
        std::ranges::sort(dofs);
        dofs.erase(std::unique(dofs.begin(), dofs.end()), dofs.end());
        std::vector<std::int32_t> physical;
        physical.reserve(dofs.size() * 3);
        for (std::int32_t d : dofs)
            for (int k = 0; k < 3; ++k)
                physical.push_back(3 * d + k);
        std::vector<fem::DirichletBC<double>> bcs;
        if (not physical.empty())
            bcs.emplace_back(0.0, physical, V_);
        return bcs;
    }

    void SolidMechanicsSolver::constrain_solution(double t)
    {
        (void)t; // a fixed boundary prescribes a constant zero displacement
        impose_values(std::span(u_->x()->array()), fixed_bcs());
    }

} // namespace hellofem::app
