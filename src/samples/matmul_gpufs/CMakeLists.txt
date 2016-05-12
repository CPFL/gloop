include_directories(
    # FIXME: Non portable code.
    /usr/local/cuda/samples/common/inc
)

add_executable(matmul_gpufs
    matrixMul.cu
    $<TARGET_OBJECTS:backward_object>
)
target_link_libraries(matmul_gpufs gloop ${GLOOP_CUDA_RUNTIME_LIBS})
set_target_properties(matmul_gpufs PROPERTIES LINKER_LANGUAGE CUDA)
