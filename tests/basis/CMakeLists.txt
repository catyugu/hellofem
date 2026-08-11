# tests/basis —— 有限元基底（hellofem::basis）测试
add_executable(test_basis
    test_lagrange.cpp
    test_elements.cpp
    test_quadrature.cpp
    test_polynomials.cpp
    test_tabulate.cpp
)
target_link_libraries(test_basis PRIVATE hellofem Catch2::Catch2WithMain)
add_test(NAME test_basis COMMAND test_basis)
