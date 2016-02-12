include_directories(
    ../../gpufs/libgpufs
    ../../gpufs/gpufs_con_lib.user
)

add_executable(copy
    fs.cu
    cp.cu
)
target_link_libraries(copy gpufs gpufs_con gloop ${GLOOP_CUDA_RUNTIME_LIBS})
set_target_properties(copy PROPERTIES LINKER_LANGUAGE CUDA)
