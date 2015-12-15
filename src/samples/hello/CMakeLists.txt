ExternalProject_Get_Property(LIBUV_LIB_EP SOURCE_DIR)
include_directories("${SOURCE_DIR}/include")

add_executable(hello
    main.cu
)
target_link_libraries(hello uv ${GLOOP_CUDA_RUNTIME_LIBS})
