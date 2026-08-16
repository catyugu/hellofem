# hellofem app tests: units, expressions, java parser, mesh loader
add_executable(app_test_units test_units.cpp)
target_link_libraries(app_test_units PRIVATE mpfem Catch2::Catch2WithMain)
add_test(NAME app_test_units COMMAND app_test_units)

add_executable(app_test_expression test_expression.cpp)
target_link_libraries(app_test_expression PRIVATE mpfem Catch2::Catch2WithMain)
add_test(NAME app_test_expression COMMAND app_test_expression)

add_executable(app_test_java_parser test_java_parser.cpp)
target_link_libraries(app_test_java_parser PRIVATE mpfem Catch2::Catch2WithMain)
add_test(NAME app_test_java_parser COMMAND app_test_java_parser)

add_executable(app_test_mesh_loader test_mesh_loader.cpp)
target_link_libraries(app_test_mesh_loader PRIVATE mpfem Catch2::Catch2WithMain)
target_compile_definitions(app_test_mesh_loader PRIVATE
    HELLOFEM_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
add_test(NAME app_test_mesh_loader COMMAND app_test_mesh_loader)

add_executable(app_test_physics test_physics.cpp)
target_link_libraries(app_test_physics PRIVATE mpfem Catch2::Catch2WithMain)
add_test(NAME app_test_physics COMMAND app_test_physics)
