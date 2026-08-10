find_package(MPI REQUIRED)
add_executable(mpi_probe mpi_probe.cpp)
target_link_libraries(mpi_probe PRIVATE MPI::MPI_CXX)
