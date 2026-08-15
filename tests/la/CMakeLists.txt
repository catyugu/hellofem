# hellofem::la tests
add_executable(test_sparsity_pattern test_sparsity_pattern.cpp)
target_link_libraries(test_sparsity_pattern PRIVATE hellofem Catch2::Catch2WithMain)
add_test(NAME test_sparsity_pattern COMMAND test_sparsity_pattern)

add_executable(test_matrix_solvers test_matrix_solvers.cpp)
target_link_libraries(test_matrix_solvers PRIVATE hellofem Catch2::Catch2WithMain)
add_test(NAME test_matrix_solvers COMMAND test_matrix_solvers)

add_executable(test_bicgstab test_bicgstab.cpp)
target_link_libraries(test_bicgstab PRIVATE hellofem Catch2::Catch2WithMain)
add_test(NAME test_bicgstab COMMAND test_bicgstab)

add_executable(test_blockmode test_blockmode.cpp)
target_link_libraries(test_blockmode PRIVATE hellofem Catch2::Catch2WithMain)
add_test(NAME test_blockmode COMMAND test_blockmode)

add_executable(test_preconditioners test_preconditioners.cpp)
target_link_libraries(test_preconditioners PRIVATE hellofem Catch2::Catch2WithMain)
add_test(NAME test_preconditioners COMMAND test_preconditioners)

add_executable(test_schwarz test_schwarz.cpp)
target_link_libraries(test_schwarz PRIVATE hellofem Catch2::Catch2WithMain)
add_test(NAME test_schwarz COMMAND test_schwarz)

add_executable(test_parallel_spmv test_parallel_spmv.cpp)
target_link_libraries(test_parallel_spmv PRIVATE hellofem Catch2::Catch2WithMain)
add_test(NAME test_parallel_spmv COMMAND test_parallel_spmv)
