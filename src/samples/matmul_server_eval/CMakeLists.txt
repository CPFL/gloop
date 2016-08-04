add_executable(matmul_server_eval
    gpu_server.cu
)
target_link_libraries(matmul_server_eval gloop ${GLOOP_CUDA_RUNTIME_LIBS})
set_target_properties(matmul_server_eval PROPERTIES LINKER_LANGUAGE CUDA)

add_executable(matmul_client_eval
    cpu_client.cc
)
target_link_libraries(matmul_client_eval
    dl
    rt
    boost_system
    boost_thread
    boost_date_time
)
