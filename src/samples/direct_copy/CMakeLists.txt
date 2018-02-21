add_executable(direct_copy
    fs.cu
    cp.cu
)
target_link_libraries(direct_copy gloop ${GLOOP_CUDA_RUNTIME_LIBS})
set_target_properties(direct_copy PROPERTIES LINKER_LANGUAGE CUDA)
