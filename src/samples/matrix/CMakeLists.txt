add_executable(matrix main.cu)
target_link_libraries(matrix ${GLOOP_CUDA_RUNTIME_LIBS})
