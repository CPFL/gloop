ExternalProject_Get_Property(LIBUV_LIB_EP SOURCE_DIR)
include_directories("${SOURCE_DIR}/include")

cuda_add_library(hooks SHARED
    gnode.cc
    main_loop.cc
    redirector.cc
)
add_dependencies(hooks uv)
target_link_libraries(hooks dl uv)
