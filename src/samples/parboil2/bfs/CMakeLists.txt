# APP_CFLAGS += -ffast-math
# APP_CXXFLAGS += -ffast-math
# APP_CUDACFLAGS += -Xptxas -dlcm=cg -arch compute_30 --use_fast_math

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ffast-math")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math -Xptxas -dlcm=cg")

add_executable(bfs-gloop
    main.cu
    ../common/src/parboil_cuda.c
)
target_link_libraries(bfs-gloop gloop ${GLOOP_CUDA_RUNTIME_LIBS})
set_target_properties(bfs-gloop PROPERTIES LINKER_LANGUAGE CUDA)
