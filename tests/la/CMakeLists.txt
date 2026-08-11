# hellofem::la tests
add_executable(test_sparsity_pattern test_sparsity_pattern.cpp)
target_link_libraries(test_sparsity_pattern PRIVATE hellofem Catch2::Catch2WithMain)
add_test(NAME test_sparsity_pattern COMMAND test_sparsity_pattern)
