include_directories(
    ../../gpufs/libgpufs
    ../../gpufs/gpufs_con_lib.user
)

add_executable(memcopy_gloop
    fs.cu
    cp.cu
)
target_link_libraries(memcopy_gloop gpufs gpufs_con gloop ${GLOOP_CUDA_RUNTIME_LIBS})
set_target_properties(memcopy_gloop PROPERTIES LINKER_LANGUAGE CUDA)
