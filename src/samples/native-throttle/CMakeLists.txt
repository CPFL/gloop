add_executable(native-throttle
    throttle.cu
)
set_target_properties(native-throttle PROPERTIES LINKER_LANGUAGE CUDA)
