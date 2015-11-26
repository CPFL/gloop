include_directories(
    "../../libgpufs"
    "../../gpufs_con_lib.user"
)

cuda_add_executable_dc(memcopy
    "${CMAKE_SOURCE_DIR}/src/gpufs/samples/memcopy/fs.cu"
    "gpufs;gpufs_con"
    "-I${CMAKE_SOURCE_DIR}/src/gpufs/libgpufs;-I${CMAKE_SOURCE_DIR}/src/gpufs/gpufs_con_lib.user"
    ""
)
