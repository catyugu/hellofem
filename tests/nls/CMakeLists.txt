# tests/nls —— hellofem::nls 非线性求解器测试
add_executable(test_newton test_newton.cpp)
target_link_libraries(test_newton PRIVATE hellofem Catch2::Catch2WithMain)
add_test(NAME test_newton COMMAND test_newton)

add_executable(test_anderson test_anderson.cpp)
target_link_libraries(test_anderson PRIVATE hellofem Catch2::Catch2WithMain)
add_test(NAME test_anderson COMMAND test_anderson)
