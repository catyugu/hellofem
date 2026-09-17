# hellofem::app tests — model-script parsing, mphtxt loading, solver solutions.
# The application layer's result verification is done with pytest
# (tests/python/test_case_results.py), not here.
add_executable(test_mpfem
    test_parsing.cpp
    test_mesh_loader.cpp
    test_physics.cpp
    test_scheduler.cpp
    test_transient.cpp
)
target_link_libraries(test_mpfem PRIVATE mpfem Catch2::Catch2WithMain)
target_compile_definitions(test_mpfem PRIVATE
    HELLOFEM_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
add_test(NAME test_mpfem COMMAND test_mpfem)
