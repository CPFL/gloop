add_executable(echo_server
    gpu_server.cu
)
target_link_libraries(echo_server gloop ${GLOOP_CUDA_RUNTIME_LIBS})
set_target_properties(echo_server PROPERTIES LINKER_LANGUAGE CUDA)

add_executable(echo_client
    gpu_client.cu
)
target_link_libraries(echo_client gloop ${GLOOP_CUDA_RUNTIME_LIBS})
set_target_properties(echo_client PROPERTIES LINKER_LANGUAGE CUDA)
