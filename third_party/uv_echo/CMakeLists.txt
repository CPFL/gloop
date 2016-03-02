ExternalProject_Get_Property(LIBUV_LIB_EP SOURCE_DIR)
include_directories("${SOURCE_DIR}/include")

add_executable(uv_echo
    main.cc
)
target_link_libraries(uv_echo
    dl
    rt
    uv
    ${CMAKE_THREAD_LIBS_INIT})
