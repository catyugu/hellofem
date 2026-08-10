# hellofem 内部测试：Catch2 + ctest，每模块一个子目录
if(NOT TARGET Catch2::Catch2WithMain)
    message(FATAL_ERROR "Catch2 target not available (see cmake/Dependencies.cmake)")
endif()

add_subdirectory(smoke)
