ExternalProject_Get_Property(LIBUV_LIB_EP SOURCE_DIR)
include_directories("${SOURCE_DIR}/include")

cuda_add_executable(hello main.cu)
add_dependencies(hello uv)
target_link_libraries(hello uv ${CMAKE_THREAD_LIBS_INIT})
