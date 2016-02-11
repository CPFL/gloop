include_directories(
    ../../gpufs/libgpufs
    ../../gpufs/gpufs_con_lib.user
)

add_executable(seqread
    seqread.cu
)
target_link_libraries(seqread gpufs gpufs_con gloop ${GLOOP_CUDA_RUNTIME_LIBS})
set_target_properties(seqread PROPERTIES LINKER_LANGUAGE CUDA)
