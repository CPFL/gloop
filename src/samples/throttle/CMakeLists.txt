add_executable(throttle
    throttle.cu
)
target_link_libraries(throttle gloop ${GLOOP_CUDA_RUNTIME_LIBS})
set_target_properties(throttle PROPERTIES LINKER_LANGUAGE CUDA)
