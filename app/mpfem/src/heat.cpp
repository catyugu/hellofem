// hellofem::app — heat transfer field solver
// SPDX-License-Identifier: MIT

#include "heat.h"

#include "kernels.h"
#include "physics_field.h"
#include "transient.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <stdexcept>

namespace hellofem::app {
    namespace {

        /// The heat-transfer physics of a case: the model's conductivity,
        /// volumetric heat capacity, heat sources and boundary conditions
        /// over the temperature, exporting T. A transient study advances it
        /// with the scheme, and it owns the Joule heating coupling.
        class HeatField : public PhysicsField {
        public:
            HeatField(const Physics& physics, CaseContext& ctx)
                : solver_(std::make_shared<HeatTransferSolver>(
                      ctx.mesh().mesh, ctx.mesh().facet_tags, ctx.mesh().cell_tags,
                      ctx.element_order(physics, "temperature")))
                , variables_ {scalar_variable("T", "(K)", solver_->solution())}
            {
                ctx.publish("T", solver_->solution());
                solver_->set_conductivity(
                    ctx.material_property("thermalconductivity"));
                // The transient heat operator needs the product of the two
                // material properties (COMSOL's volumetric heat capacity).
                solver_->set_thermal_mass(ctx.property([](const Material& material) {
                    const MaterialProperty* density = material.property("density");
                    const MaterialProperty* capacity
                        = material.property("heatcapacity");
                    if (not density or not capacity)
                        return std::string {};
                    return "(" + density->scalar_value() + ")*("
                        + capacity->scalar_value() + ")";
                }));

                // The model's features, each binding the part of the physics
                // it owns: the volumetric heat sources, the temperature
                // Dirichlet data (COMSOL 6.2 names the feature
                // TemperatureBoundary), the convective heat flux and the
                // initial values of a transient study ("Tinit").
                auto source = ctx.zero_property();
                for (const PhysicsFeature& feature : physics.features) {
                    const auto& props = feature.properties;
                    if (feature.type == "HeatSource") {
                        // A domain without a source keeps the zero one.
                        for (int dom : feature.selection)
                            source->set_expression(dom, props.at("Q0"));
                    }
                    else if (feature.type == "TemperatureBoundary"
                        or feature.type == "Temperature") {
                        for (int id : feature.selection)
                            solver_->add_temperature_bc(
                                id, ctx.expression(props.at("T0")));
                    }
                    else if (feature.type == "HeatFluxBoundary"
                        and props.contains("HeatFluxType")
                        and props.at("HeatFluxType") == "ConvectiveHeatFlux") {
                        auto h = ctx.uniform_property(props.at("h"));
                        // The ambient temperature is `Text`. The feature also
                        // carries a `minput_temperature`, but it is inert:
                        // measured on the busbar model, setting it to 100 degC
                        // leaves the solution bit-for-bit identical to the
                        // default, while `Text` moves the peak by 63 K.
                        auto t_inf = ctx.uniform_property(props.at("Text"));
                        for (int id : feature.selection)
                            solver_->add_convection(id, h, t_inf);
                    }
                    else if (feature.type == "SolidLayeredShell") {
                        // COMSOL's thin layer, thermally thin (the layer
                        // type "Conductive"): the shell conducts along the
                        // boundary and holds no temperature difference
                        // across its thickness, so the boundary carries the
                        // tangential operator ds k grad_t T . grad_t phi and
                        // nothing else. The thermally thick model
                        // ("Resistive") is a different formulation.
                        if (props.at("UserDefThicknessLayerType") != "Conductive")
                            throw std::runtime_error("heat: thin layer '"
                                + feature.tag + "' is of type '"
                                + props.at("UserDefThicknessLayerType")
                                + "', which the app does not solve");
                        if (props.at("lth_mat") != "userdef")
                            throw std::runtime_error("heat: thin layer '"
                                + feature.tag + "' takes its thickness from '"
                                + props.at("lth_mat") + "'");
                        if (props.at("k_mat") != "from_mat")
                            throw std::runtime_error("heat: thin layer '"
                                + feature.tag + "' takes its conductivity from '"
                                + props.at("k_mat") + "'");
                        // The thickness and the conductivity are separate
                        // coefficients of the operator: the conductivity is
                        // the boundary's own material (a surface material,
                        // not the domain's).
                        solver_->add_thin_layer(feature.selection,
                            ctx.uniform_property(props.at("lth")),
                            ctx.boundary_property(feature.selection,
                                [](const Material& material) {
                                    const MaterialProperty* k = material.property(
                                        "thermalconductivity");
                                    return k ? k->scalar_value() : std::string {};
                                }));
                    }
                    if (props.contains("Tinit"))
                        solver_->set_initial_temperature(
                            ctx.expression(props.at("Tinit")));
                }
                solver_->set_source(source);

                if (ctx.model().study.transient)
                    stepper_ = std::make_unique<TimeStepper>(*solver_, ctx.time_settings());
                spdlog::info("heat: bound heat transfer");
            }

            /// The Joule heating coupling: ∫ sigma |grad V|² phi joins the
            /// heat load, with the potential and the conductivity of the
            /// electric field of the case.
            void bind_couplings(CaseContext& ctx) override
            {
                const bool heating = std::any_of(ctx.model().couplings.begin(),
                    ctx.model().couplings.end(),
                    [](const MultiphysicsCoupling& coupling) {
                        return coupling.type == "ElectromagneticHeating";
                    });
                if (not heating)
                    return;
                auto potential = ctx.solution("V");
                if (not potential)
                    throw std::runtime_error("ElectromagneticHeating without an "
                                             "electric potential 'V'");
                solver_->set_joule_source(potential,
                    ctx.material_property("electricconductivity"));
            }

            TimeStepper* stepper() override { return stepper_.get(); }

            void initialize(double t0) override
            {
                if (not stepper_) {
                    // A field the study does not advance starts from the
                    // constraints alone; the level's solve provides its
                    // values.
                    solver_->constrain_solution(t0);
                    return;
                }
                // The scheme advances the model's initial values from t0.
                solver_->apply_initial_condition(t0);
                solver_->constrain_solution(t0);
                stepper_->start(t0);
            }

            void solve_level(double t) override
            {
                solver_->solve_steady(t);
                spdlog::info("heat: T solved at t = {} s", t);
            }

            std::span<const Variable> variables() const override
            {
                return variables_;
            }

        private:
            std::shared_ptr<HeatTransferSolver> solver_;
            std::unique_ptr<TimeStepper> stepper_;
            std::vector<Variable> variables_;
        };

        const FieldRegistration heat_field {"HeatTransfer",
            [](const Physics& physics, CaseContext& ctx)
                -> std::unique_ptr<PhysicsField> {
                return std::make_unique<HeatField>(physics, ctx);
            }};

    } // namespace

    // ---------------------------------------------------------------------------
    // HeatTransferSolver
    // ---------------------------------------------------------------------------

    HeatTransferSolver::HeatTransferSolver(
        std::shared_ptr<const mesh::Mesh<double>> mesh,
        std::shared_ptr<const mesh::MeshTags<int>> facet_tags,
        std::shared_ptr<const mesh::MeshTags<int>> cell_tags, int order)
        : TimeDependentField(std::move(mesh), std::move(facet_tags),
              std::move(cell_tags), order, 1)
    {
    }

    void HeatTransferSolver::set_joule_source(
        std::shared_ptr<const fem::Function<double>> V,
        std::shared_ptr<CellProperty> sigma)
    {
        joule_V_ = std::move(V);
        joule_sigma_ = std::move(sigma);
    }

    void HeatTransferSolver::refresh(double t)
    {
        t_ = t;
        if (k_)
            k_->update(t);
        if (rho_cp_)
            rho_cp_->update(t);
        if (Q_)
            Q_->update(t);
        if (joule_sigma_)
            joule_sigma_->update(t);
        for (auto& cv : convections_) {
            cv.h->update(t);
            cv.t_inf->update(t);
        }
        for (auto& layer : thin_layers_) {
            layer.ds->update(t);
            layer.k->update(t);
        }
    }

    void HeatTransferSolver::constrain_solution(double t)
    {
        impose_values(std::span(u_->x()->array()), make_bcs(temps_, t));
    }

    void HeatTransferSolver::apply_initial_condition(double t0)
    {
        const auto coords = V_->tabulate_dof_coordinates(false);
        auto& arr = u_->x()->array();
        for (std::int32_t d = 0; d < V_->dofmap()->index_map->size_local(); ++d)
            arr[static_cast<std::size_t>(d)] = initial_.eval(
                coords[3 * d], coords[3 * d + 1], coords[3 * d + 2], t0);
    }

    void HeatTransferSolver::assemble_sources(la::Vector<double>& f) const
    {
        if (Q_)
            add_load(f, {Q_->function()}, kernels::load_scalar);

        // Joule heating: ∫ sigma |grad V|² phi.
        if (joule_V_ and joule_sigma_)
            add_load(f, {joule_sigma_->function(), joule_V_},
                kernels::joule_heat_load);

        // Convection load h Tinf.
        for (const auto& cv : convections_)
            add_load(f, {cv.h->function(), cv.t_inf->function()},
                kernels::convection_load, {cv.boundary_id});
    }

    void HeatTransferSolver::assemble_step(la::MatrixCSR<double>& A,
        la::Vector<double>& b, const TimeLevel& level) const
    {
        const TimeWeights& w = level.weights;
        const double t = level.time;
        const auto& history = level.history;
        const auto& dofmap = *V_->dofmap();
        const int bs = dofmap.index_map_bs();
        const bool has_mass = w.a[0] != 0.0;

        auto bcs = make_bcs(temps_, t);
        auto rows = marked_rows(bcs);

        // Operators, assembled with the Dirichlet rows zeroed and their
        // columns kept (the boundary values move to the right-hand side in
        // impose_dirichlet).
        la::MatrixCSR<double> M(*pattern_);
        la::MatrixCSR<double> K(*pattern_);
        if (has_mass)
            add_operator(M, rows, {rho_cp_->function()}, kernels::mass_scalar);
        auto a = add_operator(K, rows, {k_->function()},
            kernels::diffusion_scalar);

        // Convection Robin mass.
        for (const auto& cv : convections_)
            add_operator(K, rows, {cv.h->function()}, kernels::convection_mass,
                {cv.boundary_id});

        // Thin layers: their own tangential conduction, which the steady
        // problem carries as well (the mass weight above is zero there).
        for (const auto& layer : thin_layers_)
            add_operator(K, rows, {layer.ds->function(), layer.k->function()},
                kernels::thin_layer_diffusion, layer.boundaries, true);

        // LHS operator: a0 M + b0 K.
        auto& av = A.values();
        if (has_mass)
            for (std::size_t i = 0; i < av.size(); ++i)
                av[i] = w.a[0] * M.values()[i] + w.b[0] * K.values()[i];
        else
            for (std::size_t i = 0; i < av.size(); ++i)
                av[i] = w.b[0] * K.values()[i];

        // Source load of this level.
        b.set(0.0);
        assemble_sources(b);

        // History: b -= (a_k M + b_k K) u^{n+1-k}.
        la::Vector<double> y(dofmap.index_map, bs);
        for (std::size_t k = 1; k < w.a.size(); ++k) {
            if (k > history.size() or history[k - 1] == nullptr)
                break;
            const la::Vector<double>& u_k = *history[k - 1];
            if (w.a[k] != 0.0 and has_mass) {
                y.set(0.0);
                M.mult(u_k, y);
                for (std::size_t i = 0; i < b.array().size(); ++i)
                    b.array()[i] -= w.a[k] * y.array()[i];
            }
            if (w.b[k] != 0.0) {
                y.set(0.0);
                K.mult(u_k, y);
                for (std::size_t i = 0; i < b.array().size(); ++i)
                    b.array()[i] -= w.b[k] * y.array()[i];
            }
        }

        impose_dirichlet(A, b, a, bcs, rows);
    }

    void HeatTransferSolver::assemble_steady(la::MatrixCSR<double>& A,
        la::Vector<double>& b) const
    {
        // The steady problem is the degenerate one-level scheme: K u = f.
        TimeWeights weights;
        weights.a = {0.0};
        weights.b = {1.0};
        la::Vector<double> f_new(V_->dofmap()->index_map,
            V_->dofmap()->index_map_bs());
        TimeLevel level;
        level.weights = weights;
        level.time = t_;
        assemble_step(A, b, level);
    }

} // namespace hellofem::app
