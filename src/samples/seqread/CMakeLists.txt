add_executable(seqread
    seqread.cu
)
target_link_libraries(seqread gloop ${GLOOP_CUDA_RUNTIME_LIBS})
set_target_properties(seqread PROPERTIES LINKER_LANGUAGE CUDA)
