include_directories(
    ../../libgpufs
    ../../gpufs_con_lib.user
)

add_executable(memcopy fs.cu)
target_link_libraries(memcopy ${GLOOP_CUDA_RUNTIME_LIBS} gpufs gpufs_con)
