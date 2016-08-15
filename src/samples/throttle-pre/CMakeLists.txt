add_executable(throttle-pre
    throttle.cu
)
target_link_libraries(throttle-pre gloop ${GLOOP_CUDA_RUNTIME_LIBS})
set_target_properties(throttle-pre PROPERTIES LINKER_LANGUAGE CUDA)
