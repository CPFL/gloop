include_directories(
    gpufs_con_lib.user
    libgpufs
)

add_subdirectory(gpufs_con_lib.user)
add_subdirectory(libgpufs)
add_subdirectory(samples)
