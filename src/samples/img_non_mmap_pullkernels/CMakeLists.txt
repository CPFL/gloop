add_executable(img_gloop_non_mmap_pullkernels
    fs.cu
    img_gpu.cu
)
target_link_libraries(img_gloop_non_mmap_pullkernels gloop ${GLOOP_CUDA_RUNTIME_LIBS})
set_target_properties(img_gloop_non_mmap_pullkernels PROPERTIES LINKER_LANGUAGE CUDA)
