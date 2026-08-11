# hellofem::la tests
add_executable(test_sparsity_pattern test_sparsity_pattern.cpp)
target_link_libraries(test_sparsity_pattern PRIVATE hellofem Catch2::Catch2WithMain)
add_test(NAME test_sparsity_pattern COMMAND test_sparsity_pattern)

add_executable(test_matrix_solvers test_matrix_solvers.cpp)
target_link_libraries(test_matrix_solvers PRIVATE hellofem Catch2::Catch2WithMain)
add_test(NAME test_matrix_solvers COMMAND test_matrix_solvers)
