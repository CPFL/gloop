add_executable(indirect
    indirect.cu
)
target_link_libraries(indirect gloop ${GLOOP_CUDA_RUNTIME_LIBS})
set_target_properties(indirect PROPERTIES LINKER_LANGUAGE CUDA)
