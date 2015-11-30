ExternalProject_Get_Property(LIBUV_LIB_EP SOURCE_DIR)
include_directories("${SOURCE_DIR}/include")

cuda_add_executable_dc(hello
    main.cu
    DEPENDS uv
    LINKER_OPTIONS ${CMAKE_THREAD_LIBS_INIT}
)
add_dependencies(hello uv)
