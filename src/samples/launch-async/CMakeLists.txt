add_executable(launch-async
    launch.cu
)
set_target_properties(launch-async PROPERTIES LINKER_LANGUAGE CUDA)
