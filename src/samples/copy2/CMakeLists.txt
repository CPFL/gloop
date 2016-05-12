add_executable(copy2
    fs.cu
    cp.cu
)
target_link_libraries(copy2 gloop ${GLOOP_CUDA_RUNTIME_LIBS})
set_target_properties(copy2 PROPERTIES LINKER_LANGUAGE CUDA)
