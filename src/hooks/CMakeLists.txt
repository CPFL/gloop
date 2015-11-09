cuda_add_library(hooks SHARED
    main_loop.cc
    redirector.cc
)
add_dependencies(hooks uv)
link_libraries(hooks dl uv)
ExternalProject_Get_Property(LIBUV_LIB_EP SOURCE_DIR)
target_include_directories(hooks PUBLIC "${SOURCE_DIR}/include")
