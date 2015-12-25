include_directories(
    ../../gpufs/libgpufs
    ../../gpufs/gpufs_con_lib.user
)

add_executable(grep_text_gloop
    fs.cu
    grep_text_gpu.cu
)
target_link_libraries(grep_text_gloop gpufs gpufs_con gloop ${GLOOP_CUDA_RUNTIME_LIBS})
set_target_properties(grep_text_gloop PROPERTIES LINKER_LANGUAGE CUDA)
