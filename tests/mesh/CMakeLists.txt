# tests/mesh —— hellofem::mesh topology/entities/connectivity tests
add_executable(test_mesh test_mesh.cpp)
target_link_libraries(test_mesh PRIVATE hellofem Catch2::Catch2WithMain)
add_test(NAME test_mesh COMMAND test_mesh)
