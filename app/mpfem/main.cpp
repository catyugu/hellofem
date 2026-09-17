// hellofem::app — mpfem application driver
// SPDX-License-Identifier: MIT

#include "case_scheduler.h"
#include "java_parser.h"
#include "mesh_loader.h"
#include "time_scheme.h"

#include "spdlog/spdlog.h"

#include <cstdio>
#include <filesystem>
#include <string>

int main(int argc, char* argv[])
{
    using namespace hellofem::app;

    TimeSettings time;
    std::string positional[3];
    int npos = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--scheme" and i + 1 < argc)
            time.scheme = argv[++i];
        else if (arg == "--tol" and i + 1 < argc)
            time.tolerance = std::stod(argv[++i]);
        else if (npos < 3)
            positional[npos++] = arg;
        else {
            std::fprintf(stderr, "unexpected argument '%s'\n", arg.c_str());
            return 2;
        }
    }
    if (npos < 3) {
        std::fprintf(stderr,
            "usage: mpfem_app <clean_model.java> <mesh.mphtxt> <result.txt> "
            "[--scheme <%s>] [--tol <time stepping tolerance>]\n",
            time_scheme_names().c_str());
        return 2;
    }

    const std::filesystem::path model_path = positional[0];
    const std::filesystem::path mesh_path = positional[1];
    const std::string result_path = positional[2];

    // Mesh.
    LoadedMesh lm = load_mphtxt_mesh(mesh_path);
    spdlog::info("mesh: {} cells, {} domains, {} boundaries, order={}",
        lm.mesh->topology()->index_map(lm.mesh->topology()->dim())->size_local(),
        lm.num_domains, lm.num_boundaries, lm.order);

    // Clean Java model script.
    ModelScript model = parse_model_java(model_path);
    spdlog::info("model '{}': {} params, {} materials, {} physics, {} couplings",
        model.name, model.parameters.size(), model.materials.size(),
        model.physics.size(), model.couplings.size());

    // Solve the model's study.
    CaseScheduler scheduler(model, lm, time);
    scheduler.run();
    scheduler.export_result(result_path);

    spdlog::info("wrote {}", result_path);
    return 0;
}
