# tests/io —— hellofem::io mesh input/output tests
add_executable(test_cells test_cells.cpp)
target_link_libraries(test_cells PRIVATE hellofem Catch2::Catch2WithMain)
add_test(NAME test_cells COMMAND test_cells)

add_executable(test_vtk test_vtk.cpp)
target_link_libraries(test_vtk PRIVATE hellofem Catch2::Catch2WithMain)
add_test(NAME test_vtk COMMAND test_vtk)

add_executable(test_mphtxt test_mphtxt.cpp)
target_compile_definitions(test_mphtxt PRIVATE
    HELLOFEM_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
target_link_libraries(test_mphtxt PRIVATE hellofem Catch2::Catch2WithMain)
add_test(NAME test_mphtxt COMMAND test_mphtxt)
