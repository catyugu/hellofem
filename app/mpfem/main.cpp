// hellofem::app — mpfem application driver (Phase F)
//
// Eats the twice-processed clean COMSOL Java model script + an mphtxt
// mesh, solves the coupled electro-thermal-structural problem and writes
// results in the same column layout as a COMSOL "Data" export, for
// direct comparison against COMSOL reference output.
//
//   mpfem_app <clean_model.java> <mesh.mphtxt> <result.txt>
//
// SPDX-License-Identifier: MIT

#include "Expression.h"
#include "java_parser.h"
#include "mesh_loader.h"
#include "physics.h"
#include "units.h"

#include "fem/Function.h"
#include "mesh/utils.h"
#include "spdlog/spdlog.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace hellofem::app {
namespace {

    /// Evaluate a material-property / BC expression (SI units). A
    /// multi-token value is a tensor array; take the diagonal (first)
    /// entry as the isotropic scalar.
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

    /// Distinct domain ids present in the mesh.
    std::set<int> domain_ids(const mesh::MeshTags<int>& cell_tags)
    {
        std::set<int> out;
        for (int v : cell_tags.values())
            out.insert(v);
        return out;
    }

    /// Value of material property `prop` on domain `dom`, if set.
    std::optional<double> prop_on_domain(const ModelScript& m, int dom,
        const std::unordered_map<std::string, double>& params,
        const std::string& prop)
    {
        const Material* mat = m.material_on_domain(dom);
        if (!mat)
            return std::nullopt;
        for (const auto& p : mat->properties)
            if (p.name == prop)
                return eval_prop(params, p.value);
        return std::nullopt;
    }

    /// Per-domain CellProperty for a material property name.
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

    /// Unit label for a COMSOL result expression (column header).
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

    /// Evaluate the requested result expressions at every mesh vertex and
    /// write a COMSOL-style "Data" export.
    void export_result(const std::string& path, const ModelScript& model,
        const std::shared_ptr<const mesh::Mesh<double>>& mesh,
        const std::shared_ptr<fem::Function<double>>& V,
        const std::shared_ptr<fem::Function<double>>& T,
        const std::shared_ptr<fem::Function<double>>& u)
    {
        auto [vc, vshape] = mesh::compute_vertex_coords(*mesh);
        const std::size_t nv = vshape[1]; // (gdim, nv)
        const std::size_t nk = model.export_config.expressions.size();
        const std::array<std::size_t, 2> xshape {nv, 3};
        std::vector<double> pts(nv * 3);
        for (std::size_t i = 0; i < nv; ++i)
            for (int q = 0; q < 3; ++q)
                pts[i * 3 + q] = vc[q * nv + i];

        std::vector<double> cols(nv * nk, 0.0);
        for (std::size_t k = 0; k < nk; ++k) {
            const std::string& name = model.export_config.expressions[k];
            if (name == "V" and V) {
                auto [ev, eshape] = V->eval(pts, xshape);
                for (std::size_t i = 0; i < nv; ++i)
                    cols[i * nk + k] = ev[i];
            }
            else if (name == "T" and T) {
                auto [ev, eshape] = T->eval(pts, xshape);
                for (std::size_t i = 0; i < nv; ++i)
                    cols[i * nk + k] = ev[i];
            }
            else if (name == "solid.disp" and u) {
                auto [ev, eshape] = u->eval(pts, xshape);
                for (std::size_t i = 0; i < nv; ++i)
                    cols[i * nk + k] = std::sqrt(ev[3 * i] * ev[3 * i]
                        + ev[3 * i + 1] * ev[3 * i + 1]
                        + ev[3 * i + 2] * ev[3 * i + 2]);
            }
            else {
                spdlog::warn("export expression '{}' not supported; writes 0", name);
            }
        }

        std::ofstream out(path);
        if (!out)
            throw std::runtime_error("cannot open result file '" + path + "'");
        // Full double precision so coordinates round-trip exactly (COMSOL
        // writes ~17 digits; shorter output breaks coordinate alignment).
        out << std::setprecision(17);
        out << "% Model:              " << model.name << ".mph\n";
        out << "% Version:            COMSOL 6.2.0.290\n";
        out << "% Dimension:          3\n";
        out << "% Nodes:              " << nv << "\n";
        out << "% Expressions:        " << nk << "\n";
        out << "% Description:        ";
        for (std::size_t k = 0; k < nk; ++k)
            out << (k ? ", " : "") << model.export_config.expressions[k];
        out << "\n";
        out << "% Length unit:        m\n";
        out << "% x                       y                        z                        ";
        for (std::size_t k = 0; k < nk; ++k)
            out << model.export_config.expressions[k] << " " << expr_unit(model.export_config.expressions[k]) << "  ";
        out << "\n";
        for (std::size_t i = 0; i < nv; ++i) {
            out << vc[0 * nv + i] << "  " << vc[1 * nv + i] << "  " << vc[2 * nv + i];
            for (std::size_t k = 0; k < nk; ++k)
                out << "  " << cols[i * nk + k];
            out << "\n";
        }
    }

} // namespace

} // namespace hellofem::app

int main(int argc, char* argv[])
{
    if (argc < 4) {
        std::fprintf(stderr,
            "usage: mpfem_app <clean_model.java> <mesh.mphtxt> <result.txt>\n");
        return 2;
    }
    using namespace hellofem::app;
    const std::filesystem::path model_path = argv[1];
    const std::filesystem::path mesh_path = argv[2];
    const std::string result_path = argv[3];

    // Mesh (mphtxt, domain/boundary ids already 1-based).
    LoadedMesh lm = load_mphtxt_mesh(mesh_path);
    spdlog::info("mesh: {} cells, {} domains, {} boundaries",
        lm.mesh->topology()->index_map(lm.mesh->topology()->dim())->size_local(),
        lm.num_domains, lm.num_boundaries);

    // Clean Java model script.
    ModelScript model = parse_model_java(model_path);
    spdlog::info("model '{}': {} params, {} materials, {} physics, {} couplings",
        model.name, model.parameters.size(), model.materials.size(),
        model.physics.size(), model.couplings.size());

    // Parameters (SI values) bound into muparser expressions.
    std::unordered_map<std::string, double> params;
    for (const auto& p : model.parameters)
        params[p.name] = p.si;

    const int order = 1;

    // ---- Electric currents -> V ----
    std::shared_ptr<CellProperty> sigma;
    std::shared_ptr<ElectrostaticsSolver> es;
    if (model.physics_by_type("ConductiveMedia")) {
        es = std::make_shared<ElectrostaticsSolver>(
            lm.mesh, lm.facet_tags, lm.cell_tags, order);
        sigma = make_property(lm.mesh, lm.cell_tags, params, model,
            "electricconductivity");
        es->set_conductivity(sigma);
        for (const auto* f : model.features("Terminal")) {
            if (f->properties.contains("V0")) {
                const double v0 = eval_prop(params, f->properties.at("V0"));
                for (int id : f->selection)
                    es->add_voltage_bc(id, v0);
            }
        }
        for (const auto* f : model.features("Ground"))
            for (int id : f->selection)
                es->add_voltage_bc(id, 0.0);
        es->solve_steady();
        spdlog::info("electric: V solved");
    }

    // ---- Heat transfer -> T (Joule source from V) ----
    std::shared_ptr<HeatTransferSolver> ht;
    if (model.physics_by_type("HeatTransfer")) {
        ht = std::make_shared<HeatTransferSolver>(
            lm.mesh, lm.facet_tags, lm.cell_tags, order);
        ht->set_conductivity(make_property(lm.mesh, lm.cell_tags, params, model,
            "thermalconductivity"));
        const bool joule = es
            and std::any_of(model.couplings.begin(), model.couplings.end(),
                [](const MultiphysicsCoupling& c) {
                    return c.type == "ElectromagneticHeating";
                });
        if (joule)
            ht->set_joule_source(es->solution(), sigma);
        for (const auto* f : model.features("Temperature")) {
            const double t0 = eval_prop(params, f->properties.at("T0"));
            for (int id : f->selection)
                ht->add_temperature_bc(id, t0);
        }
        for (const auto* f : model.features("HeatFluxBoundary")) {
            for (int id : f->selection)
                std::fprintf(stderr, " %d", id);
            std::fprintf(stderr, " }\n");
            const std::string type = f->properties.contains("HeatFluxType")
                ? f->properties.at("HeatFluxType")
                : "HeatFlux";
            if (type != "ConvectiveHeatFlux")
                continue;
            const double h = eval_prop(params, f->properties.at("h"));
            const double tinf = f->properties.contains("minput_temperature")
                ? eval_prop(params, f->properties.at("minput_temperature"))
                : 0.0;
            for (int id : f->selection)
                ht->add_convection(id, h, tinf);
        }
        ht->solve_steady();
        spdlog::info("heat: T solved");
    }

    // ---- Solid mechanics -> u (thermal expansion from T) ----
    std::shared_ptr<SolidMechanicsSolver> sm;
    if (model.physics_by_type("SolidMechanics")) {
        sm = std::make_shared<SolidMechanicsSolver>(
            lm.mesh, lm.facet_tags, lm.cell_tags, order);
        sm->set_elastic(
            make_property(lm.mesh, lm.cell_tags, params, model, "E"),
            make_property(lm.mesh, lm.cell_tags, params, model, "nu"));
        if (ht) {
            for (const auto& c : model.couplings)
                if (c.type == "ThermalExpansion") {
                    double tref = 293.15;
                    if (c.properties.contains("minput_strainreferencetemperature"))
                        tref = eval_prop(params, c.properties.at("minput_strainreferencetemperature"));
                    auto alpha = make_property(lm.mesh, lm.cell_tags, params, model,
                        "thermalexpansioncoefficient");
                    sm->set_thermal_expansion(ht->solution(), alpha, tref);
                }
        }
        for (const auto* f : model.features("Fixed"))
            for (int id : f->selection)
                sm->add_fixed_bc(id);
        sm->solve_steady();
        spdlog::info("solid: u solved");
    }

    // ---- Export COMSOL-consistent result ----
    export_result(result_path, model, lm.mesh, es ? es->solution() : nullptr,
        ht ? ht->solution() : nullptr, sm ? sm->solution() : nullptr);
    spdlog::info("wrote {}", result_path);
    return 0;
}
