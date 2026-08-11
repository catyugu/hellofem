# tests/common —— hellofem::common 模块测试
add_executable(test_common test_common.cpp)
target_link_libraries(test_common PRIVATE hellofem Catch2::Catch2WithMain)
add_test(NAME test_common COMMAND test_common)
