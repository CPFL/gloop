add_executable(launch
    launch.cu
)
target_link_libraries(launch gloop ${GLOOP_CUDA_RUNTIME_LIBS})
set_target_properties(launch PROPERTIES LINKER_LANGUAGE CUDA)
