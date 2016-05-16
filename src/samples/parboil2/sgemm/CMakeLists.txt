set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ffast-math")

add_executable(sgemm
    io.cc
    main.cu
    ../common/src/parboil_cuda.c
)
set_target_properties(sgemm PROPERTIES LINKER_LANGUAGE CUDA)
