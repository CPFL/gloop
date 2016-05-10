add_executable(throttle2
    throttle.cu
)
target_link_libraries(throttle2 gloop ${GLOOP_CUDA_RUNTIME_LIBS})
set_target_properties(throttle2 PROPERTIES LINKER_LANGUAGE CUDA)
