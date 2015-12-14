include_directories(
    ../../gpufs/libgpufs
    ../../gpufs/gpufs_con_lib.user
)

add_executable(memcopy_manual
    fs.cu
    cp.cu
)
target_link_libraries(memcopy_manual ${CMAKE_THREAD_LIBS_INIT} gpufs gpufs_con gloop)
