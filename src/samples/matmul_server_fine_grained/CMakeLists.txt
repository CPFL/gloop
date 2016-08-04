add_executable(matmul_server_fine_grained
    gpu_server.cu
)
target_link_libraries(matmul_server_fine_grained gloop ${GLOOP_CUDA_RUNTIME_LIBS})
set_target_properties(matmul_server_fine_grained PROPERTIES LINKER_LANGUAGE CUDA)

add_executable(matmul_client_fine_grained
    cpu_client.cc
)
target_link_libraries(matmul_client_fine_grained
    dl
    rt
    boost_system
    boost_thread
    boost_date_time
)
