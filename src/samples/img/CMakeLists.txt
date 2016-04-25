add_executable(img_gloop
    fs.cu
    img_gpu.cu
)
target_link_libraries(img_gloop gloop ${GLOOP_CUDA_RUNTIME_LIBS})
set_target_properties(img_gloop PROPERTIES LINKER_LANGUAGE CUDA)
