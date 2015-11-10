cuda_add_executable(grep_text main.cu)
target_link_libraries(grep_text uv gloop ${CMAKE_THREAD_LIBS_INIT})
