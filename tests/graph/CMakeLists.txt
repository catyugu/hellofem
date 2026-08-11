# tests/graph —— hellofem::graph 模块测试
add_executable(test_graph test_graph.cpp)
target_link_libraries(test_graph PRIVATE hellofem Catch2::Catch2WithMain)
add_test(NAME test_graph COMMAND test_graph)
