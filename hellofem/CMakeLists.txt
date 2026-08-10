# hellofem 库（单库结构，保持与原骨架一致：GLOB 收集所有模块源文件）
file(GLOB_RECURSE HELLOFEM_SOURCES CONFIGURE_DEPENDS *.cpp *.cc *.c)
file(GLOB_RECURSE HELLOFEM_HEADERS CONFIGURE_DEPENDS *.h *.hpp)

add_library(hellofem STATIC ${HELLOFEM_SOURCES} ${HELLOFEM_HEADERS})

# include 根：hellofem/ 目录（头文件按 <mod>/xxx.h 引用，如 basis/types.h、common/IndexMap.h）
target_include_directories(hellofem PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    ${CMAKE_BINARY_DIR}   # config.h
)

# Eigen 头目录标记为 SYSTEM，避免 /W4 /WX 对 Eigen 头报警告。
# Eigen3::Eigen 在 Eigen 5.0 中为 ALIAS target（不能 set_target_properties）；
# 且 Eigen 为 header-only，直接取其 include/definitions 而非链接该 target。
if(TARGET Eigen3::Eigen)
    get_target_property(_eigen_incs Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
    get_target_property(_eigen_defs Eigen3::Eigen INTERFACE_COMPILE_DEFINITIONS)
    target_include_directories(hellofem SYSTEM PUBLIC ${_eigen_incs})
    if(_eigen_defs)
        target_compile_definitions(hellofem PUBLIC ${_eigen_defs})
    endif()
endif()

target_link_libraries(hellofem PUBLIC hellofem_options spdlog::spdlog MPI::MPI_CXX)

target_compile_options(hellofem PRIVATE
    $<$<CXX_COMPILER_FRONTEND_VARIANT:GNU>:-Werror>
)

install(TARGETS hellofem EXPORT hellofemTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES ${HELLOFEM_HEADERS}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/hellofem
)
