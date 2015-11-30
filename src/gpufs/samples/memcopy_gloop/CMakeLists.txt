include_directories(
    ../../libgpufs
    ../../gpufs_con_lib.user
)

cuda_add_executable_dc(memcopy_gloop_manual
    "fs.cu;cp.cu"
    "gpufs;gpufs_con"
    ""
    ""
)
