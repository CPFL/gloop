add_executable(loop
    launch.cu
)
set_target_properties(loop PROPERTIES LINKER_LANGUAGE CUDA)
