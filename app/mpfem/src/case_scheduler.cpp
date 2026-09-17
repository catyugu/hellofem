// hellofem::app — case-level solver scheduler implementation
// SPDX-License-Identifier: MIT

#include "case_scheduler.h"

#include "solver.h"

#include "Expression.h"
#include "mesh/utils.h"
#include "spdlog/spdlog.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <set>
#include <stdexcept>

namespace hellofem::app {
    namespace {

        const char* expr_unit(const std::string& name)
        {
            if (name == "V")
                return "(V)";
            if (name == "T")
                return "(K)";
            if (name == "solid.disp")
                return "(m)";
            if (name == "ec.normJ")
                return "(A/m^2)";
            if (name == "ec.Qh")
                return "(W/m^3)";
            if (name == "solid.mises")
                return "(N/m^2)";
            return "(1)";
        }

        std::set<int> domain_ids(const mesh::MeshTags<int>& cell_tags)
        {
            std::set<int> out;
            for (int v : cell_tags.values())
                out.insert(v);
            return out;
        }

        /// The material property `name` on `material`, if it defines one.
        const MaterialProperty* material_property(const Material& material,
            const std::string& name)
        {
            for (const auto& p : material.properties)
                if (p.name == name)
                    return &p;
            return nullptr;
        }

        /// A material property value as a scalar expression. COMSOL stores a
        /// tensor-valued property as a space-separated component list; the
        /// isotropic case uses the leading component.
        std::string scalar_value(const MaterialProperty& property)
        {
            const std::size_t space = property.value.find(' ');
            return space == std::string::npos ? property.value
                                              : property.value.substr(0, space);
        }

    } // namespace

    // =========================================================================
    // CaseScheduler
    // =========================================================================

    CaseScheduler::CaseScheduler(const ModelScript& model, const LoadedMesh& lm)
        : model_(model)
        , lm_(lm)
        , mesh_(lm.mesh)
        , facet_tags_(lm.facet_tags)
        , cell_tags_(lm.cell_tags)
        , order_(lm.order)
    {
        for (const auto& p : model_.parameters)
            params_[p.name] = p.si;
    }

    void CaseScheduler::set_time_scheme(std::string_view name)
    {
        scheme_name_ = std::string(name);
    }

    ScalarExpression CaseScheduler::expr(std::string_view text) const
    {
        return ScalarExpression(text, params_);
    }

    // -------------------------------------------------------------------------
    // Model binding
    // -------------------------------------------------------------------------

    std::shared_ptr<CellProperty> CaseScheduler::make_property(
        const std::string& name)
    {
        auto property = std::make_shared<CellProperty>(mesh_, cell_tags_, params_);
        // A material expression may read the solution: bind the fields first,
        // so a field symbol shadows a parameter of the same name.
        if (T_)
            property->bind_field("T", T_);
        if (V_)
            property->bind_field("V", V_);
        for (int dom : domain_ids(*cell_tags_))
            if (const Material* material = model_.material_on_domain(dom))
                if (const MaterialProperty* p = material_property(*material, name))
                    property->set_expression(dom, scalar_value(*p));
        return property;
    }

    std::shared_ptr<CellProperty> CaseScheduler::make_thermal_mass()
    {
        // The transient heat operator needs the product of the two material
        // properties (COMSOL's volumetric heat capacity).
        auto property = std::make_shared<CellProperty>(mesh_, cell_tags_, params_);
        if (T_)
            property->bind_field("T", T_);
        for (int dom : domain_ids(*cell_tags_)) {
            const Material* material = model_.material_on_domain(dom);
            if (!material)
                continue;
            const MaterialProperty* density = material_property(*material, "density");
            const MaterialProperty* cp = material_property(*material, "heatcapacity");
            if (density and cp)
                property->set_expression(dom,
                    "(" + scalar_value(*density) + ")*(" + scalar_value(*cp) + ")");
        }
        return property;
    }

    std::shared_ptr<CellProperty> CaseScheduler::make_boundary_property(
        std::string_view text)
    {
        // A boundary coefficient is one expression on every cell (the facet
        // kernels read the coefficient of the adjacent cell).
        auto property = std::make_shared<CellProperty>(mesh_, cell_tags_, params_);
        if (T_)
            property->bind_field("T", T_);
        for (int dom : domain_ids(*cell_tags_))
            property->set_expression(dom, text);
        return property;
    }

    void CaseScheduler::bind_electric()
    {
        if (not model_.physics_by_type("ConductiveMedia"))
            return;
        es_ = std::make_shared<ElectrostaticsSolver>(mesh_, facet_tags_,
            cell_tags_, order_);
        V_ = es_->solution();
        sigma_ = make_property("electricconductivity");
        es_->set_conductivity(sigma_);

        for (const auto* f : model_.features("Terminal"))
            if (f->properties.contains("V0"))
                for (int id : f->selection)
                    es_->add_voltage_bc(id, expr(f->properties.at("V0")));
        for (const auto* f : model_.features("Ground"))
            for (int id : f->selection)
                es_->add_voltage_bc(id, ScalarExpression(0.0));
        spdlog::info("electric: bound conductive media");
    }

    void CaseScheduler::bind_heat()
    {
        const Physics* heat = model_.physics_by_type("HeatTransfer");
        if (not heat)
            return;
        ht_ = std::make_shared<HeatTransferSolver>(mesh_, facet_tags_,
            cell_tags_, order_);
        T_ = ht_->solution();
        ht_->set_conductivity(make_property("thermalconductivity"));
        ht_->set_thermal_mass(make_thermal_mass());

        const bool joule = std::any_of(model_.couplings.begin(),
            model_.couplings.end(), [](const MultiphysicsCoupling& c) {
                return c.type == "ElectromagneticHeating";
            });
        if (joule and V_ and sigma_)
            ht_->set_joule_source(V_, sigma_);

        // Volumetric heat sources on the domains of a HeatSource feature.
        auto source = std::make_shared<CellProperty>(mesh_, cell_tags_, params_);
        if (T_)
            source->bind_field("T", T_);
        for (const auto* f : model_.features("HeatSource"))
            if (f->properties.contains("Q0"))
                for (int dom : f->selection)
                    source->set_expression(dom, f->properties.at("Q0"));
        ht_->set_source(source);

        // Temperature Dirichlet data (COMSOL 6.2 names it TemperatureBoundary).
        for (const char* type : {"TemperatureBoundary", "Temperature"})
            for (const auto* f : model_.features(type))
                if (f->properties.contains("T0"))
                    for (int id : f->selection)
                        ht_->add_temperature_bc(id, expr(f->properties.at("T0")));

        // Convective heat flux.
        for (const auto* f : model_.features("HeatFluxBoundary")) {
            const auto& props = f->properties;
            if (props.contains("HeatFluxType")
                and props.at("HeatFluxType") != "ConvectiveHeatFlux")
                continue;
            if (not props.contains("h"))
                continue;
            auto h = make_boundary_property(props.at("h"));
            auto t_inf = make_boundary_property(props.contains("minput_temperature")
                    ? props.at("minput_temperature")
                    : "0");
            for (int id : f->selection)
                ht_->add_convection(id, h, t_inf);
        }

        // Initial values of a transient study ("Tinit").
        for (const auto& feature : heat->features)
            if (feature.properties.contains("Tinit"))
                ht_->set_initial_temperature(expr(feature.properties.at("Tinit")));
        spdlog::info("heat: bound heat transfer");
    }

    void CaseScheduler::bind_solid()
    {
        if (not model_.physics_by_type("SolidMechanics"))
            return;
        sm_ = std::make_shared<SolidMechanicsSolver>(mesh_, facet_tags_,
            cell_tags_, order_);
        u_ = sm_->solution();
        sm_->set_elastic(make_property("E"), make_property("nu"));

        if (T_)
            for (const auto& c : model_.couplings)
                if (c.type == "ThermalExpansion") {
                    // COMSOL's heat-transfer reference temperature default.
                    double t_ref = 293.15;
                    if (c.properties.contains("minput_strainreferencetemperature"))
                        t_ref = expr(c.properties.at("minput_strainreferencetemperature"))
                                    .eval(0, 0, 0, 0);
                    sm_->set_thermal_expansion(T_,
                        make_property("thermalexpansioncoefficient"), t_ref);
                }

        for (const auto* f : model_.features("Fixed"))
            for (int id : f->selection)
                sm_->add_fixed_bc(id);
        spdlog::info("solid: bound solid mechanics");
    }

    // -------------------------------------------------------------------------
    // Time levels
    // -------------------------------------------------------------------------

    void CaseScheduler::solve_electric(double t)
    {
        if (not es_)
            return;
        solve_system(
            [&](la::MatrixCSR<double>& A, la::Vector<double>& b) {
                es_->refresh(t);
                es_->assemble_steady(A, b);
            },
            *V_->x(), es_->pattern(), es_->nonlinear());
        spdlog::info("electric: V solved at t = {} s", t);
    }

    void CaseScheduler::solve_heat(double t)
    {
        if (not ht_)
            return;
        solve_system(
            [&](la::MatrixCSR<double>& A, la::Vector<double>& b) {
                ht_->refresh(t);
                ht_->assemble_steady(A, b);
            },
            *T_->x(), ht_->pattern(), ht_->nonlinear());
        spdlog::info("heat: T solved at t = {} s", t);
    }

    void CaseScheduler::solve_solid(double t)
    {
        if (not sm_)
            return;
        solve_system(
            [&](la::MatrixCSR<double>& A, la::Vector<double>& b) {
                sm_->refresh(t);
                sm_->assemble_steady(A, b);
            },
            *u_->x(), sm_->pattern(), sm_->nonlinear());
        spdlog::info("solid: u solved at t = {} s", t);
    }

    void CaseScheduler::run()
    {
        bind_electric();
        bind_heat();
        bind_solid();
        if (model_.study.transient)
            run_transient();
        else
            run_stationary();
    }

    void CaseScheduler::run_stationary()
    {
        solve_electric(0.0);
        solve_heat(0.0);
        solve_solid(0.0);
        record(0.0);
    }

    void CaseScheduler::run_transient()
    {
        const auto& times = model_.study.times;
        if (times.size() < 2)
            throw std::runtime_error(
                "CaseScheduler: a transient study needs at least two output times");
        if (ht_)
            stepper_ = std::make_unique<HeatTimeStepper>(ht_,
                make_time_scheme(scheme_name_));
        if (stepper_)
            spdlog::info("transient: {} output times, scheme '{}' (order {})",
                times.size(), stepper_->scheme().name(),
                stepper_->scheme().order());

        // Initial state: the initial values of the heat equation (the
        // evolutionary field) and the solution of the algebraic, quasi-static
        // fields at t0, which the constraints alone do not determine.
        const double t0 = times.front();
        if (ht_)
            ht_->apply_initial_condition();
        FieldSolver* fields[] = {es_.get(), ht_.get(), sm_.get()};
        for (FieldSolver* field : fields)
            if (field)
                field->constrain_solution(t0);
        solve_electric(t0);
        solve_solid(t0);
        if (stepper_)
            stepper_->start(t0);
        record(t0);

        for (std::size_t i = 1; i < times.size(); ++i) {
            solve_electric(times[i]);
            if (stepper_)
                stepper_->step(times[i]);
            solve_solid(times[i]);
            record(times[i]);
        }
    }

    // -------------------------------------------------------------------------
    // Result export
    // -------------------------------------------------------------------------

    std::vector<double> CaseScheduler::evaluate_columns() const
    {
        auto [vc, vshape] = mesh::compute_vertex_coords(*mesh_);
        const std::size_t nv = vshape[1];
        const std::size_t nk = model_.export_config.expressions.size();
        const std::array<std::size_t, 2> shape {nv, 3};
        std::vector<double> columns(nv * nk, 0.0);

        // The vertex coordinates are component-major; the evaluator wants
        // point-major rows.
        std::vector<double> pts(nv * 3);
        for (std::size_t i = 0; i < nv; ++i)
            for (int q = 0; q < 3; ++q)
                pts[i * 3 + static_cast<std::size_t>(q)] = vc[q * nv + i];

        for (std::size_t k = 0; k < nk; ++k) {
            const std::string& name = model_.export_config.expressions[k];
            if (name == "V" and V_) {
                auto [values, vshadow] = V_->eval(pts, shape);
                for (std::size_t i = 0; i < nv; ++i)
                    columns[i * nk + k] = values[i];
            }
            else if (name == "T" and T_) {
                auto [values, vshadow] = T_->eval(pts, shape);
                for (std::size_t i = 0; i < nv; ++i)
                    columns[i * nk + k] = values[i];
            }
            else if (name == "solid.disp" and u_) {
                auto [values, vshadow] = u_->eval(pts, shape);
                for (std::size_t i = 0; i < nv; ++i)
                    columns[i * nk + k] = std::sqrt(values[3 * i] * values[3 * i]
                        + values[3 * i + 1] * values[3 * i + 1]
                        + values[3 * i + 2] * values[3 * i + 2]);
            }
            else {
                spdlog::warn("export '{}' not supported; writes 0", name);
            }
        }
        return columns;
    }

    void CaseScheduler::record(double t)
    {
        Snapshot snapshot;
        snapshot.time = t;
        snapshot.columns = evaluate_columns();
        snapshots_.push_back(std::move(snapshot));
    }

    void CaseScheduler::export_result(const std::string& path) const
    {
        auto [vc, vshape] = mesh::compute_vertex_coords(*mesh_);
        const std::size_t nv = vshape[1];
        const std::size_t nk = model_.export_config.expressions.size();
        const std::size_t nt = snapshots_.size();
        if (nt == 0)
            throw std::runtime_error("CaseScheduler: no result to export");
        const bool per_time = model_.study.transient;

        std::ofstream out(path);
        if (!out)
            throw std::runtime_error("cannot open " + path);
        out << std::setprecision(17);
        out << "% Model:              " << model_.name << ".mph\n";
        out << "% Version:            COMSOL 6.2.0.290\n";
        out << "% Dimension:          3\n";
        out << "% Nodes:              " << nv << "\n";
        out << "% Expressions:        " << nk * nt << "\n";
        out << "% Description:        ";
        for (std::size_t k = 0; k < nk; ++k)
            out << (k ? ", " : "") << model_.export_config.expressions[k];
        out << "\n% Length unit:        m\n";
        out << "% x                       y                        z                        ";
        for (const Snapshot& snapshot : snapshots_)
            for (std::size_t k = 0; k < nk; ++k) {
                out << model_.export_config.expressions[k] << " "
                    << expr_unit(model_.export_config.expressions[k]);
                if (per_time)
                    out << " @ t=" << snapshot.time;
                out << "  ";
            }
        out << "\n";
        for (std::size_t i = 0; i < nv; ++i) {
            out << vc[0 * nv + i] << "  " << vc[1 * nv + i] << "  " << vc[2 * nv + i];
            for (const Snapshot& snapshot : snapshots_)
                for (std::size_t k = 0; k < nk; ++k)
                    out << "  " << snapshot.columns[i * nk + k];
            out << "\n";
        }
    }

} // namespace hellofem::app
