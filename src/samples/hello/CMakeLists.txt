cuda_add_executable(hello main.cu)
add_dependencies(hello uv)
ExternalProject_Get_Property(LIBUV_LIB_EP SOURCE_DIR)
target_include_directories(hooks PUBLIC "${SOURCE_DIR}/include")
