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
            {
                // A material law of this physics reads the field's own
                // solution: publish it before the coefficients are built.
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
                for (const PhysicsFeature& feature : physics.features) {
                    if (feature.type == "HeatSource") {
                        // A source belongs to the domains its own feature
                        // selects; a domain no feature selects carries none,
                        // and a model with no source term carries none at all.
                        solver_->add_source(feature.selection,
                            ctx.uniform_property(feature.required("Q0")));
                    }
                    else if (feature.type == "TemperatureBoundary"
                        or feature.type == "Temperature") {
                        for (int id : feature.selection)
                            solver_->add_temperature_bc(
                                id, ctx.expression(feature.required("T0")));
                    }
                    else if (feature.type == "HeatFluxBoundary") {
                        const std::string& flux = feature.required("HeatFluxType");
                        if (flux != "ConvectiveHeatFlux")
                            throw std::runtime_error("heat: the heat flux '"
                                + feature.tag + "' is of type '" + flux
                                + "', which the app does not solve");
                        // The ambient temperature is `Text`. The feature also
                        // carries a `minput_temperature`, but it is inert:
                        // measured on the busbar model, setting it to 100 degC
                        // leaves the solution bit-for-bit identical to the
                        // default, while `Text` moves the peak by 63 K.
                        for (int id : feature.selection)
                            solver_->add_convection(
                                ctx.facet_property(feature.required("h"), id),
                                ctx.facet_property(feature.required("Text"), id));
                    }
                    else if (feature.type == "SolidLayeredShell") {
                        // COMSOL's thin layer, thermally thin (the layer
                        // type "Conductive"): the shell conducts along the
                        // boundary and holds no temperature difference
                        // across its thickness, so the boundary carries the
                        // tangential operator ds k grad_t T . grad_t phi and
                        // nothing else. The thermally thick model
                        // ("Resistive") is a different formulation.
                        const std::string& layer_type
                            = feature.required("UserDefThicknessLayerType");
                        if (layer_type != "Conductive")
                            throw std::runtime_error("heat: thin layer '"
                                + feature.tag + "' is of type '" + layer_type
                                + "', which the app does not solve");
                        const std::string& thickness_from
                            = feature.required("lth_mat");
                        if (thickness_from != "userdef")
                            throw std::runtime_error("heat: thin layer '"
                                + feature.tag + "' takes its thickness from '"
                                + thickness_from + "'");
                        const std::string& conductivity_from
                            = feature.required("k_mat");
                        if (conductivity_from != "from_mat")
                            throw std::runtime_error("heat: thin layer '"
                                + feature.tag + "' takes its conductivity from '"
                                + conductivity_from + "'");
                        // The thickness and the conductivity are separate
                        // coefficients of the operator, and the conductivity
                        // is the material COMSOL applies on the boundary —
                        // a surface material, which may differ from the one
                        // of the domain it borders and from one boundary of
                        // the selection to the next. A facet integral reads
                        // its coefficient off the adjacent cell, so one
                        // integral carries one value: one integral per
                        // boundary, each with its own boundary's material.
                        for (int id : feature.selection) {
                            const Material* material
                                = ctx.model().material_on_boundary(id);
                            if (material == nullptr)
                                throw std::runtime_error("heat: thin layer '"
                                    + feature.tag + "' sits on boundary "
                                    + std::to_string(id)
                                    + ", which carries no material");
                            const MaterialProperty* k
                                = material->property("thermalconductivity");
                            if (k == nullptr)
                                throw std::runtime_error("heat: thin layer '"
                                    + feature.tag + "': material '"
                                    + material->tag
                                    + "' defines no thermalconductivity");
                            solver_->add_thin_layer(
                                ctx.facet_property(feature.required("lth"), id),
                                ctx.facet_property(k->scalar_value(), id));
                        }
                    }
                    else if (feature.type == "Init" or feature.type.empty()) {
                        if (feature.properties.contains("Tinit"))
                            solver_->set_initial_temperature(
                                ctx.expression(feature.required("Tinit")));
                    }
                    else
                        throw std::runtime_error("heat: the feature '"
                            + feature.tag + "' is of type '" + feature.type
                            + "', which the app does not solve");
                }

                // The state the rest of the case reads this field at: the one
                // the stepper samples, so that a coupling or a result at a
                // time between two levels sees the field there — and
                // reporting one never writes the solution the stepping is at.
                if (ctx.model().study.transient)
                    stepper_ = std::make_unique<TimeStepper>(*solver_, ctx.time_settings());
                const std::shared_ptr<const fem::Function<double>> state
                    = stepper_ ? stepper_->sampled() : solver_->solution();
                variables_ = {scalar_variable("T", "(K)", state)};
                ctx.publish("T", state);
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

        const FieldRegistration heat_field {"HeatTransfer", heat_time_tolerance,
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

    void HeatTransferSolver::add_source(const std::set<int>& domains,
        std::shared_ptr<DomainProperty> Q)
    {
        // COMSOL lets a later feature override an earlier one on the same
        // domain. The app integrates a source per feature, so two features
        // sharing a domain would add where the reference replaces: a model it
        // cannot solve is refused rather than summed.
        for (const Source& source : sources_)
            for (int dom : domains)
                if (source.domains.contains(dom))
                    throw std::runtime_error("heat: domain "
                        + std::to_string(dom)
                        + " carries two heat sources, which the app does not "
                          "solve");
        sources_.push_back({domains, std::move(Q)});
    }

    void HeatTransferSolver::add_convection(std::shared_ptr<FacetProperty> h,
        std::shared_ptr<FacetProperty> t_inf)
    {
        // One integral carries the data of one boundary: the coefficients of
        // another boundary would leave this one at zero.
        if (h->boundary() != t_inf->boundary())
            throw std::runtime_error("heat: convection with h on boundary "
                + std::to_string(h->boundary()) + " and Text on boundary "
                + std::to_string(t_inf->boundary()));
        convections_.push_back({std::move(h), std::move(t_inf)});
    }

    void HeatTransferSolver::add_thin_layer(std::shared_ptr<FacetProperty> ds,
        std::shared_ptr<FacetProperty> k)
    {
        if (ds->boundary() != k->boundary())
            throw std::runtime_error("heat: thin layer with ds on boundary "
                + std::to_string(ds->boundary()) + " and k on boundary "
                + std::to_string(k->boundary()));
        thin_layers_.push_back({std::move(ds), std::move(k)});
    }

    void HeatTransferSolver::set_joule_source(
        std::shared_ptr<const fem::Function<double>> V,
        std::shared_ptr<DomainProperty> sigma)
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
        for (auto& source : sources_)
            source.Q->update(t);
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
        // A source is integrated over the domains it belongs to.
        for (const Source& source : sources_)
            add_load(f, {source.Q->function()}, kernels::load_scalar,
                source.domains);

        // Joule heating: ∫ sigma |grad V|² phi.
        if (joule_V_ and joule_sigma_)
            add_load(f, {joule_sigma_->function(), joule_V_},
                kernels::joule_heat_load);

        // Convection load h Tinf.
        for (const auto& cv : convections_)
            add_load(f, {cv.h->function(), cv.t_inf->function()},
                kernels::convection_load, {cv.h->boundary()});
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
                {cv.h->boundary()});

        // Thin layers: their own tangential conduction, which the steady
        // problem carries as well (the mass weight above is zero there).
        for (const auto& layer : thin_layers_)
            add_operator(K, rows, {layer.ds->function(), layer.k->function()},
                kernels::thin_layer_diffusion, {layer.ds->boundary()}, true);

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
