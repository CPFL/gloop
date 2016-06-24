add_executable(shared
    throttle.cu
)
target_link_libraries(shared gloop ${GLOOP_CUDA_RUNTIME_LIBS})
set_target_properties(shared PROPERTIES LINKER_LANGUAGE CUDA)
