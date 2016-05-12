add_executable(copy
    fs.cu
    cp.cu
)
target_link_libraries(copy gloop ${GLOOP_CUDA_RUNTIME_LIBS})
set_target_properties(copy PROPERTIES LINKER_LANGUAGE CUDA)
