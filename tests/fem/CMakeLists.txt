# tests/fem —— hellofem::fem 核心（CoordinateElement/ElementDofLayout）+ cell_types 测试
add_executable(test_fem_core test_fem_core.cpp)
target_link_libraries(test_fem_core PRIVATE hellofem Catch2::Catch2WithMain)
add_test(NAME test_fem_core COMMAND test_fem_core)

add_executable(test_finite_element test_finite_element.cpp)
target_link_libraries(test_finite_element PRIVATE hellofem Catch2::Catch2WithMain)
add_test(NAME test_finite_element COMMAND test_finite_element)

add_executable(test_dofmap test_dofmap.cpp)
target_link_libraries(test_dofmap PRIVATE hellofem Catch2::Catch2WithMain)
add_test(NAME test_dofmap COMMAND test_dofmap)

add_executable(test_sparsitybuild test_sparsitybuild.cpp)
target_link_libraries(test_sparsitybuild PRIVATE hellofem Catch2::Catch2WithMain)
add_test(NAME test_sparsitybuild COMMAND test_sparsitybuild)
