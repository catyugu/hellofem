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

add_executable(test_function_space test_function_space.cpp)
target_link_libraries(test_function_space PRIVATE hellofem Catch2::Catch2WithMain)
add_test(NAME test_function_space COMMAND test_function_space)

add_executable(test_constant_function test_constant_function.cpp)
target_link_libraries(test_constant_function PRIVATE hellofem Catch2::Catch2WithMain)
add_test(NAME test_constant_function COMMAND test_constant_function)

add_executable(test_dirichletbc test_dirichletbc.cpp)
target_link_libraries(test_dirichletbc PRIVATE hellofem Catch2::Catch2WithMain)
add_test(NAME test_dirichletbc COMMAND test_dirichletbc)

add_executable(test_form test_form.cpp)
target_link_libraries(test_form PRIVATE hellofem Catch2::Catch2WithMain)
add_test(NAME test_form COMMAND test_form)

add_executable(test_poisson test_poisson.cpp)
target_link_libraries(test_poisson PRIVATE hellofem Catch2::Catch2WithMain)
add_test(NAME test_poisson COMMAND test_poisson)

add_executable(test_interpolate test_interpolate.cpp)
target_link_libraries(test_interpolate PRIVATE hellofem Catch2::Catch2WithMain)
add_test(NAME test_interpolate COMMAND test_interpolate)

add_executable(test_eval test_eval.cpp)
target_link_libraries(test_eval PRIVATE hellofem Catch2::Catch2WithMain)
add_test(NAME test_eval COMMAND test_eval)
