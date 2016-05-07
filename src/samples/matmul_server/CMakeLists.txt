add_executable(matmul_server
    gpu_server.cu
)
target_link_libraries(matmul_server gloop ${GLOOP_CUDA_RUNTIME_LIBS})
set_target_properties(matmul_server PROPERTIES LINKER_LANGUAGE CUDA)
