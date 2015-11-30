include_directories(
    ../../libgpufs
    ../../gpufs_con_lib.user
)

cuda_add_executable_dc(memcopy
    fs.cu
    DEPENDS gpufs;gpufs_con
)
