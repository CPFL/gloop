add_executable(proportional-throttle
    throttle.cu
)
target_link_libraries(proportional-throttle gloop ${GLOOP_CUDA_RUNTIME_LIBS})
set_target_properties(proportional-throttle PROPERTIES LINKER_LANGUAGE CUDA)
