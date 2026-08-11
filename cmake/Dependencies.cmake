# =============================================================================
# Dependencies.cmake - 全部外部依赖引入
#
# 全部经 CPM 引入（学 mpfem 的 cmake/Dependencies.cmake 范式）：
#   spdlog / Eigen(GitLab 5.0) / muparser / Catch2 / oneTBB / amgcl
# 无 MPI（单进程多线程，并行由 oneTBB 提供）。
# =============================================================================

include(CPM)

# 依赖以静态库构建，避免运行时 DLL 查找问题（spdlog/Catch2/muparser/TBB）。
set(BUILD_SHARED_LIBS OFF)

# ---------------------------------------------------------------------------
# 1. spdlog（日志）
# ---------------------------------------------------------------------------
CPMAddPackage(
    NAME spdlog
    GITHUB_REPOSITORY gabime/spdlog
    GIT_TAG v1.17.0
    OPTIONS
    "SPDLOG_BUILD_SHARED OFF"
    "SPDLOG_BUILD_EXAMPLE OFF"
    "SPDLOG_BUILD_EXAMPLE_HOUSE OFF"
    "SPDLOG_BUILD_TESTS OFF"
    "SPDLOG_BUILD_BENCH OFF"
)

# ---------------------------------------------------------------------------
# 2. Eigen 5.0（GitLab，用户确认该仓库有 5.0.x）
# ---------------------------------------------------------------------------
CPMAddPackage(
    NAME Eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 5.0.1
    OPTIONS
    "BUILD_TESTING OFF"
    "EIGEN_BUILD_DOC OFF"
    "EIGEN_BUILD_PKGCONFIG OFF"
)

# Eigen 头目录标记为 SYSTEM 的处理放在 hellofem/CMakeLists.txt：
# Eigen 5.0 中 Eigen3::Eigen 是 ALIAS target，无法在此 set_target_properties。

# ---------------------------------------------------------------------------
# 3. muparser（表达式解析，应用层使用）
# ---------------------------------------------------------------------------
CPMAddPackage(
    NAME muparser
    GITHUB_REPOSITORY beltoforion/muparser
    GIT_TAG v2.3.5
    OPTIONS
    "MUPARSER_BUILD_SAMPLES OFF"
    "MUPARSER_BUILD_EXAMPLES OFF"
    "MUPARSER_BUILD_TESTS OFF"
    "MUPARSER_WITH_OPENMP OFF"
)

# ---------------------------------------------------------------------------
# 4. Catch2（内部库测试）
# ---------------------------------------------------------------------------
CPMAddPackage(
    NAME Catch2
    GITHUB_REPOSITORY catchorg/Catch2
    GIT_TAG v3.15.3
    OPTIONS
    "CATCH_BUILD_TESTING OFF"
    "CATCH_BUILD_EXAMPLES OFF"
    "CATCH_INSTALL_DOCS OFF"
)

# ---------------------------------------------------------------------------
# 5. oneTBB（多线程并行，替代 MPI 的并行能力）
# ---------------------------------------------------------------------------
CPMAddPackage(
    NAME oneTBB
    GITHUB_REPOSITORY oneapi-src/oneTBB
    GIT_TAG v2023.1.0
    OPTIONS
    "TBB_TEST OFF"
    "TBB_EXAMPLES OFF"
    "TBB_BENCH OFF"
    "TBB_STRICT OFF"
    "TBB_ENABLE_IPO OFF"
    "TBB_BUILD_STATIC ON"
    "TBB_BUILD_SHARED OFF"
    "TBB_NO_COPY_GENERATOR ON"
)

# ---------------------------------------------------------------------------
# 6. amgcl（header-only AMG 预条件器；la 模块 Phase 6 使用）
#    CPM 下 AMGCL_MASTER_PROJECT=OFF，不构建其 tests/examples；Boost/MPI 均为
#    可选查找，缺失时 amgcl 自动定义 AMGCL_NO_BOOST。
# ---------------------------------------------------------------------------
CPMAddPackage(
    NAME amgcl
    GITHUB_REPOSITORY ddemidov/amgcl
    GIT_TAG 1.5.0
)
