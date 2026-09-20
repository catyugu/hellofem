// hellofem::app — mpfem application driver
// SPDX-License-Identifier: MIT

#include "case_scheduler.h"
#include "java_parser.h"
#include "mesh_loader.h"
#include "physics_field.h"
#include "time_scheme.h"

#include "spdlog/spdlog.h"

#include <cstdio>
#include <filesystem>
#include <string>

int main(int argc, char* argv[])
{
    using namespace hellofem::app;

    std::string positional[3];
    int npos = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (npos < 3)
            positional[npos++] = arg;
        else {
            std::fprintf(stderr, "unexpected argument '%s'\n", arg.c_str());
            return 2;
        }
    }
    if (npos < 3) {
        std::fprintf(stderr,
            "usage: mpfem_app <clean_model.java> <mesh.mphtxt> <result.txt>\n");
        return 2;
    }

    const std::filesystem::path model_path = positional[0];
    const std::filesystem::path mesh_path = positional[1];
    const std::string result_path = positional[2];

    // Clean Java model script. It is read first: it states the unit the
    // geometry — and so the mesh — is built in.
    ModelScript model = parse_model_java(model_path);
    spdlog::info("model '{}': {} params, {} materials, {} physics, {} couplings",
        model.name, model.parameters.size(), model.materials.size(),
        model.physics.size(), model.couplings.size());

    // Mesh, converted into SI through the model's geometry length unit.
    LoadedMesh lm = load_mphtxt_mesh(mesh_path, model.length_unit);
    spdlog::info("mesh: {} cells, {} domains, {} boundaries, order={}",
        lm.mesh->topology()->index_map(lm.mesh->topology()->dim())->size_local(),
        lm.num_domains, lm.num_boundaries, lm.order);

    // The time stepping is the model's own: the scheme and the order are the
    // defaults the reference's solver runs at (see `TimeSettings`), and the
    // accuracy its steps are held to is the reference's own physics-controlled
    // tolerance — the tightest the model's physics interfaces recommend, or
    // the tolerance a study step that states one asks for, which is what its
    // reference solution was computed with.
    TimeSettings time;
    time.tolerance = model.study.tolerance
        ? *model.study.tolerance
        : study_time_tolerance(model.physics);

    // Solve the model's study. A failing solve is reported where it happens:
    // an exception that escapes `main` ends the process without its message,
    // which reads as a crash rather than as the solver's own complaint.
    try {
        CaseScheduler scheduler(model, lm, time);
        scheduler.run();
        scheduler.export_result(result_path);
    }
    catch (const std::exception& e) {
        spdlog::error("{}", e.what());
        return 1;
    }

    spdlog::info("wrote {}", result_path);
    return 0;
}
