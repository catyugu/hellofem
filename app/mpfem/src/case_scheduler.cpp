// hellofem::app — case-level solver scheduler implementation
// SPDX-License-Identifier: MIT

#include "case_scheduler.h"

#include "Expression.h"
#include "basis/element-families.h"
#include "fem/DirichletBC.h"
#include "fem/assembler.h"
#include "fem/dofmapbuilder.h"
#include "fem/precompute.h"
#include "fem/sparsitybuild.h"
#include "kernels.h"
#include "la/KrylovSolver.h"
#include "la/SparsityPattern.h"
#include "la/Vector.h"
#include "mesh/Topology.h"
#include "mesh/cell_types.h"
#include "mesh/utils.h"
#include "nls/AndersonPicard.h"
#include "spdlog/spdlog.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <set>
#include <span>

namespace hellofem::app {
namespace {

    double eval_prop(const std::unordered_map<std::string, double>& params,
        const std::string& text)
    {
        const std::string first = text.substr(0, text.find(' '));
        std::unordered_map<std::string, double*> vars;
        std::vector<double> storage;
        storage.reserve(params.size());
        for (const auto& [k, v] : params) {
            storage.push_back(v);
            vars[k] = &storage.back();
        }
        Expression e;
        e.parse(first, vars);
        return e.eval(0, 0, 0, 0);
    }

    std::set<int> domain_ids(const mesh::MeshTags<int>& cell_tags)
    {
        std::set<int> out;
        for (int v : cell_tags.values())
            out.insert(v);
        return out;
    }

    std::optional<double> prop_on_domain(const ModelScript& m, int dom,
        const std::unordered_map<std::string, double>& params,
        const std::string& prop)
    {
        const Material* mat = m.material_on_domain(dom);
        if (!mat) return std::nullopt;
        for (const auto& p : mat->properties)
            if (p.name == prop)
                return eval_prop(params, p.value);
        return std::nullopt;
    }

    std::shared_ptr<CellProperty> make_property(
        const std::shared_ptr<const mesh::Mesh<double>>& mesh,
        const std::shared_ptr<const mesh::MeshTags<int>>& cell_tags,
        const std::unordered_map<std::string, double>& params,
        const ModelScript& m, const std::string& prop)
    {
        auto cp = std::make_shared<CellProperty>(mesh, cell_tags);
        for (int dom : domain_ids(*cell_tags)) {
            if (auto v = prop_on_domain(m, dom, params, prop))
                cp->set_domain(dom, *v);
        }
        return cp;
    }

    const char* expr_unit(const std::string& name)
    {
        if (name == "V") return "(V)";
        if (name == "T") return "(K)";
        if (name == "solid.disp") return "(m)";
        if (name == "ec.normJ") return "(A/m^2)";
        if (name == "ec.Qh") return "(W/m^3)";
        if (name == "solid.mises") return "(N/m^2)";
        return "(1)";
    }


    /// Locate the dofs on the given boundary facets.

} // anonymous namespace

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

void CaseScheduler::run()
{
    solve_electric();
    solve_heat();
    solve_solid();
}

// -------------------------------------------------------------------------
// Electric currents
// -------------------------------------------------------------------------

void CaseScheduler::solve_electric()
{
    if (!model_.physics_by_type("ConductiveMedia"))
        return;

    es_ = std::make_shared<ElectrostaticsSolver>(
        mesh_, facet_tags_, cell_tags_, order_);

    auto sigma = make_property(mesh_, cell_tags_, params_, model_,
        "electricconductivity");
    ht_sigma_ = sigma;
    es_->set_conductivity(sigma);

    for (const auto* f : model_.features("Terminal")) {
        if (f->properties.contains("V0")) {
            const double v0 = eval_prop(params_, f->properties.at("V0"));
            for (int id : f->selection)
                es_->add_voltage_bc(id, v0);
        }
    }
    for (const auto* f : model_.features("Ground"))
        for (int id : f->selection)
            es_->add_voltage_bc(id, 0.0);

    es_->solve_steady();
    V_ = es_->solution();
    spdlog::info("electric: V solved");
}

// -------------------------------------------------------------------------
// Heat transfer
// -------------------------------------------------------------------------

void CaseScheduler::solve_heat()
{
    if (!model_.physics_by_type("HeatTransfer"))
        return;

    ht_ = std::make_shared<HeatTransferSolver>(
        mesh_, facet_tags_, cell_tags_, order_);

    ht_joule_ = V_ && std::any_of(model_.couplings.begin(),
        model_.couplings.end(), [](const MultiphysicsCoupling& c) {
            return c.type == "ElectromagneticHeating";
        });

    // Nonlinear k(T) or constant? Check alpha_k to decide.
    const auto* akp = model_.parameter("alpha_k");
    if (akp) {
        spdlog::info("heat: Picard path (k(T) nonlinear)");
        // k(T) = k0 * (1 + alpha_k * (T - Tref))
        const double k0val = model_.parameter("k0")->si;
        const double akval = akp->si;
        const double tref  = model_.parameter("Tref")->si;

        auto cp_k = std::make_shared<CellProperty>(mesh_, cell_tags_);
        const std::size_t nc = static_cast<std::size_t>(
            mesh_->topology()->index_map(mesh_->topology()->dim())->size_local());
        for (std::size_t c = 0; c < nc; ++c)
            (*cp_k->function()->x())[c] = k0val;
        ht_->set_conductivity(cp_k);

        fill_heat_bcs();
        solve_heat_picard(k0val, akval, tref, cp_k);
    }
    else {
        spdlog::info("heat: linear path");
        auto cp_k = make_property(mesh_, cell_tags_, params_, model_,
            "thermalconductivity");
        ht_->set_conductivity(cp_k);
        fill_heat_bcs();
        ht_->solve_steady();
    }

    T_ = ht_->solution();
    spdlog::info("heat: T solved");
}

void CaseScheduler::solve_heat_picard(double k0, double ak, double tref,
    std::shared_ptr<CellProperty> cp_k)
{
    const std::size_t nc = static_cast<std::size_t>(
        mesh_->topology()->index_map(mesh_->topology()->dim())->size_local());

    std::vector<std::int32_t> cells(nc);
    std::iota(cells.begin(), cells.end(), 0);

    la::SparsityPattern pattern(ht_->space()->dofmap()->index_map, 1);
    fem::sparsitybuild::cells(pattern, std::pair{cells, cells},
        {*ht_->space()->dofmap(), *ht_->space()->dofmap()});
    std::vector<std::int32_t> diag(
        ht_->space()->dofmap()->index_map->size_local());
    std::iota(diag.begin(), diag.end(), 0);
    pattern.insert_diagonal(std::span(diag));
    pattern.finalize();

    auto& karr = cp_k->function()->x()->array();
    ht_->solution()->x()->set(300.0);

    nls::AndersonConfig cfg;
    cfg.depth = 5;
    cfg.warmup_iters = 3;
    cfg.relative_tolerance = 1e-8;
    cfg.absolute_tolerance = 1e-12;
    cfg.max_iterations = 100;
    cfg.preconditioner_type = "jacobi";

    auto result = nls::anderson_picard<double>(
        [&](const la::Vector<double>& x)
            -> std::pair<la::MatrixCSR<double>, la::Vector<double>> {
            for (std::size_t c = 0; c < nc; ++c) {
                auto dofs = ht_->space()->dofmap()->cell_dofs(
                    static_cast<std::int32_t>(c));
                double Tavg = 0;
                for (auto d : dofs)
                    Tavg += x[static_cast<std::size_t>(d)];
                Tavg /= static_cast<double>(dofs.size());
                karr[c] = k0 * (1.0 + ak * (Tavg - tref));
            }
            la::MatrixCSR<double> A(pattern);
            la::Vector<double> b(ht_->space()->dofmap()->index_map,
                ht_->space()->dofmap()->index_map_bs());
            ht_->assemble_system(A, b);
            return {std::move(A), std::move(b)};
        },
        *ht_->solution()->x(), cfg);

    if (!result.converged)
        throw std::runtime_error(
            "CaseScheduler: Picard heat solve did not converge");
    spdlog::info("heat (Picard): {} iterations, {} Krylov iters",
        result.iterations, result.krylov_iterations);
}

void CaseScheduler::fill_heat_bcs()
{
    // Joule heating source (if coupled).
    if (ht_joule_)
        ht_->set_joule_source(V_, ht_sigma_);

    // Temperature Dirichlet BCs (COMSOL 6.2 uses "TemperatureBoundary").
    // Check both "Temperature" (older COMSOL) and "TemperatureBoundary".
    auto apply_temp = [&](const std::string& type) {
        for (const auto* f : model_.features(type)) {
            const double t0 = eval_prop(params_, f->properties.at("T0"));
            for (int id : f->selection)
                ht_->add_temperature_bc(id, t0);
        }
    };
    apply_temp("Temperature");
    apply_temp("TemperatureBoundary");

    // Convective heat flux BCs.
    for (const auto* f : model_.features("HeatFluxBoundary")) {
        const auto& props = f->properties;
        const std::string type = props.contains("HeatFluxType")
            ? props.at("HeatFluxType") : "HeatFlux";
        if (type != "ConvectiveHeatFlux")
            continue;
        const double h = eval_prop(params_, props.at("h"));
        const double tinf = props.contains("minput_temperature")
            ? eval_prop(params_, props.at("minput_temperature"))
            : 0.0;
        for (int id : f->selection)
            ht_->add_convection(id, h, tinf);
    }
}

// -------------------------------------------------------------------------
// Solid mechanics
// -------------------------------------------------------------------------

void CaseScheduler::solve_solid()
{
    if (!model_.physics_by_type("SolidMechanics"))
        return;

    sm_ = std::make_shared<SolidMechanicsSolver>(
        mesh_, facet_tags_, cell_tags_, order_);

    sm_->set_elastic(
        make_property(mesh_, cell_tags_, params_, model_, "E"),
        make_property(mesh_, cell_tags_, params_, model_, "nu"));

    if (T_) {
        for (const auto& c : model_.couplings)
            if (c.type == "ThermalExpansion") {
                double tref = 293.15;
                if (c.properties.contains("minput_strainreferencetemperature"))
                    tref = eval_prop(params_,
                        c.properties.at("minput_strainreferencetemperature"));
                auto alpha = make_property(mesh_, cell_tags_, params_, model_,
                    "thermalexpansioncoefficient");
                sm_->set_thermal_expansion(T_, alpha, tref);
            }
    }

    for (const auto* f : model_.features("Fixed"))
        for (int id : f->selection)
            sm_->add_fixed_bc(id);

    sm_->solve_steady();
    u_ = sm_->solution();
    spdlog::info("solid: u solved");
}

// -------------------------------------------------------------------------
// Result export
// -------------------------------------------------------------------------

void CaseScheduler::export_result(const std::string& path) const
{
    auto [vc, vshape] = mesh::compute_vertex_coords(*mesh_);
    const std::size_t nv = vshape[1];
    const std::size_t nk = model_.export_config.expressions.size();
    const std::array<std::size_t, 2> xshape{nv, 3};
    std::vector<double> pts(nv * 3);
    for (std::size_t i = 0; i < nv; ++i)
        for (int q = 0; q < 3; ++q)
            pts[i * 3 + q] = vc[q * nv + i];

    std::vector<double> cols(nv * nk, 0.0);
    for (std::size_t k = 0; k < nk; ++k) {
        const std::string& name = model_.export_config.expressions[k];
        if (name == "V" && V_) {
            auto [ev, eshape] = V_->eval(pts, xshape);
            for (std::size_t i = 0; i < nv; ++i)
                cols[i * nk + k] = ev[i];
        }
        else if (name == "T" && T_) {
            auto [ev, eshape] = T_->eval(pts, xshape);
            for (std::size_t i = 0; i < nv; ++i)
                cols[i * nk + k] = ev[i];
        }
        else if (name == "solid.disp" && u_) {
            auto [ev, eshape] = u_->eval(pts, xshape);
            for (std::size_t i = 0; i < nv; ++i)
                cols[i * nk + k] = std::sqrt(ev[3 * i] * ev[3 * i]
                    + ev[3 * i + 1] * ev[3 * i + 1]
                    + ev[3 * i + 2] * ev[3 * i + 2]);
        }
        else {
            spdlog::warn("export '{}' not supported; writes 0", name);
        }
    }

    std::ofstream out(path);
    if (!out)
        throw std::runtime_error("cannot open " + path);
    out << std::setprecision(17);
    out << "% Model:              " << model_.name << ".mph\n";
    out << "% Version:            COMSOL 6.2.0.290\n";
    out << "% Dimension:          3\n";
    out << "% Nodes:              " << nv << "\n";
    out << "% Expressions:        " << nk << "\n";
    out << "% Description:        ";
    for (std::size_t k = 0; k < nk; ++k)
        out << (k ? ", " : "") << model_.export_config.expressions[k];
    out << "\n% Length unit:        m\n";
    out << "% x                       y                        z                        ";
    for (std::size_t k = 0; k < nk; ++k)
        out << model_.export_config.expressions[k] << " "
            << expr_unit(model_.export_config.expressions[k]) << "  ";
    out << "\n";
    for (std::size_t i = 0; i < nv; ++i) {
        out << vc[0 * nv + i] << "  " << vc[1 * nv + i] << "  " << vc[2 * nv + i];
        for (std::size_t k = 0; k < nk; ++k)
            out << "  " << cols[i * nk + k];
        out << "\n";
    }
}

} // namespace hellofem::app