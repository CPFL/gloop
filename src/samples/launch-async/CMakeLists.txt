add_executable(launch-async
    launch.cu
)
target_link_libraries(launch-async ${GLOOP_CUDA_RUNTIME_LIBS})
set_target_properties(launch-async PROPERTIES LINKER_LANGUAGE CUDA)
