# tests/fem —— hellofem::fem 核心（CoordinateElement/ElementDofLayout）+ cell_types 测试
add_executable(test_fem_core test_fem_core.cpp)
target_link_libraries(test_fem_core PRIVATE hellofem Catch2::Catch2WithMain)
add_test(NAME test_fem_core COMMAND test_fem_core)
