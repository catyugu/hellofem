# tools/mpi_probe —— MPI 可用性探针
# 由 tools/CMakeLists.txt 以子目录引入；也可独立配置（单独 find_package(MPI)）。
# 运行：mpiexec -n 4 build/tools/mpi_probe.exe
if(NOT TARGET MPI::MPI_CXX)
    find_package(MPI REQUIRED)
endif()

add_executable(mpi_probe mpi_probe.cpp)

if(TARGET hellofem_options)
    target_link_libraries(mpi_probe PRIVATE MPI::MPI_CXX hellofem_options)
else()
    target_link_libraries(mpi_probe PRIVATE MPI::MPI_CXX)
endif()
