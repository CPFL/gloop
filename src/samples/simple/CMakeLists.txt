include_directories(
    ../../gpufs/libgpufs
    ../../gpufs/gpufs_con_lib.user
)

add_executable(simple
    fs.cu
    cp.cu
    # $<TARGET_OBJECTS:backward_object>
)
target_link_libraries(simple gpufs gpufs_con gloop ${GLOOP_CUDA_RUNTIME_LIBS})
set_target_properties(simple PROPERTIES LINKER_LANGUAGE CUDA)
