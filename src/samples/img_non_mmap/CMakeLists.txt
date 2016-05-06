add_executable(img_gloop_non_mmap
    fs.cu
    img_gpu.cu
)
target_link_libraries(img_gloop_non_mmap gloop ${GLOOP_CUDA_RUNTIME_LIBS})
set_target_properties(img_gloop_non_mmap PROPERTIES LINKER_LANGUAGE CUDA)
