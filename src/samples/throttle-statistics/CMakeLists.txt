add_executable(throttle-statistics
    throttle.cu
)
target_link_libraries(throttle-statistics gloop ${GLOOP_CUDA_RUNTIME_LIBS})
set_target_properties(throttle-statistics PROPERTIES LINKER_LANGUAGE CUDA)
