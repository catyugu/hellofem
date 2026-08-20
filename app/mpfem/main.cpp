// hellofem::app — mpfem application driver (Phase F)
// SPDX-License-Identifier: MIT

#include "case_scheduler.h"
#include "java_parser.h"
#include "mesh_loader.h"

#include "spdlog/spdlog.h"

#include <cstdio>
#include <filesystem>
#include <string>

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

    // Dispatch and solve.
    CaseScheduler sched(model, lm);
    sched.run();
    sched.export_result(result_path);

    spdlog::info("wrote {}", result_path);
    return 0;
}
