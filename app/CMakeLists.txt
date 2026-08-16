# hellofem application layer: mpfem (COMSOL-Java-driven electro-thermal-structural solver)

# ---- mpfem library (app code, links hellofem + muparser) ----
file(GLOB_RECURSE MPFEM_SOURCES CONFIGURE_DEPENDS mpfem/src/*.cpp)
file(GLOB_RECURSE MPFEM_HEADERS CONFIGURE_DEPENDS mpfem/src/*.h)

add_library(mpfem STATIC ${MPFEM_SOURCES} ${MPFEM_HEADERS})
target_include_directories(mpfem PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/mpfem/src)
target_link_libraries(mpfem PUBLIC hellofem muparser::muparser)

# ---- mpfem application executable ----
add_executable(mpfem_app mpfem/main.cpp)
target_link_libraries(mpfem_app PRIVATE mpfem)

# ---- app-level Catch2 tests ----
option(HELLOFEM_BUILD_APP_TESTS "build app/mpfem unit tests" ON)
if(HELLOFEM_BUILD_APP_TESTS)
    add_subdirectory(mpfem/tests)
endif()
