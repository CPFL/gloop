find_package(OpenMP REQUIRED)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
# Short term solution.
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler ${OpenMP_CXX_FLAGS}")

add_executable(lavaMD-gloop
    main.c
    kernel/kernel_gpu_cuda_wrapper.cu
    util/device/device.cu
    util/num/num.c
    util/timer/timer.c
)
set_target_properties(lavaMD-gloop PROPERTIES LINKER_LANGUAGE CUDA)
target_link_libraries(lavaMD-gloop gloop ${GLOOP_CUDA_RUNTIME_LIBS})
