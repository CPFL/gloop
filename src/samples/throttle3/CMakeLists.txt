add_executable(throttle3
    throttle.cu
)
target_link_libraries(throttle3 gloop ${GLOOP_CUDA_RUNTIME_LIBS})
set_target_properties(throttle3 PROPERTIES LINKER_LANGUAGE CUDA)
