include_directories(
    ../../gpufs/libgpufs
    ../../gpufs/gpufs_con_lib.user
    # FIXME: Non portable code.
    /usr/local/cuda/samples/common/inc
)

add_executable(matmul
    matrixMul.cu
    $<TARGET_OBJECTS:backward_object>
)
target_link_libraries(matmul gpufs gpufs_con gloop ${GLOOP_CUDA_RUNTIME_LIBS})
set_target_properties(matmul PROPERTIES LINKER_LANGUAGE CUDA)
