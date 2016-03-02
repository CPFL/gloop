include_directories(
    ../../gpufs/libgpufs
    ../../gpufs/gpufs_con_lib.user
)

add_executable(matmul_server
    gpu_server.cu
)
target_link_libraries(matmul_server gpufs gpufs_con gloop ${GLOOP_CUDA_RUNTIME_LIBS})
set_target_properties(matmul_server PROPERTIES LINKER_LANGUAGE CUDA)

add_executable(matmul_client
    gpu_client.cu
)
target_link_libraries(matmul_client gpufs gpufs_con gloop ${GLOOP_CUDA_RUNTIME_LIBS})
set_target_properties(matmul_client PROPERTIES LINKER_LANGUAGE CUDA)
