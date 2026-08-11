# tests/geometry —— hellofem::geometry（GJK / BoundingBoxTree / 查询）
add_executable(test_geometry test_geometry.cpp)
target_link_libraries(test_geometry PRIVATE hellofem Catch2::Catch2WithMain)
add_test(NAME test_geometry COMMAND test_geometry)
