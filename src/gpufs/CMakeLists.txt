if (CMAKE_BUILD_TYPE MATCHES Release)
    add_definitions(
        -DNDEBUG
        -DDEBUG_NOINLINE=
        -DRELEASE=1
        -DMALLOC_STATS
    )
endif()

if (CMAKE_BUILD_TYPE MATCHES Debug)
    add_definitions(
        -DDEBUG_NOINLINE=__noinline__
        -DMALLOC_STATS
    )
endif()

include_directories(
    "gpufs_con_lib.user"
    "libgpufs"
)

add_subdirectory(gpufs_con_lib.user)
add_subdirectory(libgpufs)
add_subdirectory(samples)
