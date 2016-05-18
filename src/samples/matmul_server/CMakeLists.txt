add_executable(matmul_server
    gpu_server.cu
)
target_link_libraries(matmul_server gloop ${GLOOP_CUDA_RUNTIME_LIBS})
set_target_properties(matmul_server PROPERTIES LINKER_LANGUAGE CUDA)

add_executable(matmul_client
    cpu_client.cc
)
target_link_libraries(matmul_client
    dl
    rt
    boost_system
    boost_thread
    boost_date_time
)
