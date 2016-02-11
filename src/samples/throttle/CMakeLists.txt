include_directories(
    ../../gpufs/libgpufs
    ../../gpufs/gpufs_con_lib.user
)

add_executable(throttle
    throttle.cu
)
target_link_libraries(throttle gpufs gpufs_con gloop ${GLOOP_CUDA_RUNTIME_LIBS})
set_target_properties(throttle PROPERTIES LINKER_LANGUAGE CUDA)
