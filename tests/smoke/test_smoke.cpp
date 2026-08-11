#include <catch2/catch_test_macros.hpp>

#include <Eigen/Dense>
#include <mpi.h>
#include <spdlog/spdlog.h>

#include <string>

// hellofem 模块占位符号：证明模块静态库可链接。
namespace hellofem::common {
    const char* module_id();
}
namespace hellofem::la {
    const char* module_id();
}

TEST_CASE("smoke: deps + module static libs link", "[smoke]")
{
    // Eigen
    Eigen::Matrix2d m;
    m << 1.0, 2.0, 3.0, 4.0;
    REQUIRE(m.sum() == 10.0);

    // spdlog
    spdlog::info("smoke: spdlog ok, Eigen sum = {}", m.sum());

    // hellofem 模块静态库链接
    REQUIRE(std::string(hellofem::common::module_id()) == "common");
    REQUIRE(std::string(hellofem::la::module_id()) == "la");

    // MPI 仅验证链接（运行时验证由 tools/mpi_probe 经 mpiexec 完成）
    auto* allreduce = &MPI_Allreduce;
    REQUIRE(allreduce != nullptr);
}
