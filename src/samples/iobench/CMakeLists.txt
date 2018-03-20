add_executable(iobench
    iobench.cu
)
target_link_libraries(iobench gloop ${GLOOP_CUDA_RUNTIME_LIBS})
set_target_properties(iobench PROPERTIES LINKER_LANGUAGE CUDA)
