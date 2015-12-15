include_directories(
    ../../libgpufs
    ../../gpufs_con_lib.user
)

add_executable(memcopy
    fs.cu
)
target_link_libraries(memcopy gpufs gpufs_con ${GLOOP_CUDA_RUNTIME_LIBS})
set_target_properties(memcopy PROPERTIES LINKER_LANGUAGE CUDA)
