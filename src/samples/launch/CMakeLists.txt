add_executable(launch
    launch.cu
)
set_target_properties(launch PROPERTIES LINKER_LANGUAGE CUDA)
