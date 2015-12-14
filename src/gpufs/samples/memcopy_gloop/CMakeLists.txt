include_directories(
    ../../libgpufs
    ../../gpufs_con_lib.user
)

add_executable(memcopy_gloop_manual
    fs.cu
    cp.cu
)
target_link_libraries(memcopy_gloop_manual ${GLOOP_CUDA_RUNTIME_LIBS} gpufs gpufs_con)
