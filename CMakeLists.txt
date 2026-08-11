cmake_minimum_required(VERSION 3.16)
project(hellofem VERSION 1.0.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(HELLOFEM_BUILD_TESTS "build tests" ON)
option(HELLOFEM_BUILD_TOOLS "build tools" ON)

# ---- CMake 模块 / 依赖 / 目标宏 ----
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(CPM_SOURCE_CACHE "${CMAKE_SOURCE_DIR}/.cache/cpm" CACHE PATH "CPM cache to avoid repeated fetching")
include(CPM) # cmake/CPM.cmake（vendor）
include(Dependencies) # 全部外部依赖经 CPM 引入（无 MPI）
include(cmake/hellofemOptions.cmake)

configure_file(${CMAKE_SOURCE_DIR}/cmake/config.h.in ${CMAKE_BINARY_DIR}/config.h)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
    message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
endif()

# 自动探测 CPU 核数，设定并行编译任务数。
cmake_host_system_information(RESULT CPU_NUMS QUERY NUMBER_OF_LOGICAL_CORES)
message(STATUS "CPU NUMS on current device is:${CPU_NUMS}")
math(EXPR JOBS "${CPU_NUMS} + ${CPU_NUMS} / 2")
set(CMAKE_BUILD_PARALLEL_LEVEL ${JOBS} CACHE STRING "parallel compiling thread num")

add_subdirectory(hellofem)
add_subdirectory(app)

if(HELLOFEM_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(HELLOFEM_BUILD_TOOLS)
    add_subdirectory(tools)
endif()
