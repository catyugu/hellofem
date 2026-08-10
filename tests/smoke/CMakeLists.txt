# 冒烟测试：验证工具链 + spdlog/Eigen/MPI 链接 + hellofem 库链接
add_executable(test_smoke test_smoke.cpp)
target_link_libraries(test_smoke PRIVATE hellofem Catch2::Catch2WithMain)
add_test(NAME test_smoke COMMAND test_smoke)
