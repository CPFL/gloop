include_directories(
    ../../gpufs/libgpufs
    ../../gpufs/gpufs_con_lib.user
)

cuda_add_executable_dc(memcopy_manual
    fs.cu
    cp.cu
    DEPENDS gpufs gpufs_con gloop
)
