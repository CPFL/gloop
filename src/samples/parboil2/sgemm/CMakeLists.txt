set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ffast-math")

add_executable(sgemm-gloop
    io.cc
    main.cu
    ../common/src/parboil_cuda.c
)
target_link_libraries(sgemm-gloop gloop ${GLOOP_CUDA_RUNTIME_LIBS})
set_target_properties(sgemm-gloop PROPERTIES LINKER_LANGUAGE CUDA)
