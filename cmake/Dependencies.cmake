# =============================================================================
# Dependencies.cmake - 全部外部依赖引入
#
# - MPI：必需依赖（find_package(MPI REQUIRED)，无单进程回退）
# - spdlog / Eigen(GitLab 5.0) / muparser / Catch2：全部经 CPM 引入
#   （学 mpfem 的 cmake/Dependencies.cmake 范式）
# =============================================================================

include(CPM)

# 依赖以静态库构建，避免运行时 DLL 查找问题（spdlog/Catch2/muparser）。
set(BUILD_SHARED_LIBS OFF)

# ---------------------------------------------------------------------------
# 1. MPI（必需）
# ---------------------------------------------------------------------------
# MS-MPI 位于 conda env 的 Library 时，配置需传
#   -DCMAKE_PREFIX_PATH=<conda-env>/Library
find_package(MPI REQUIRED)

# ---------------------------------------------------------------------------
# 2. spdlog（日志）
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
# 3. Eigen 5.0（GitLab，用户确认该仓库有 5.0.x）
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
# 4. muparser（表达式解析，应用层使用）
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
# 5. Catch2（内部库测试）
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
