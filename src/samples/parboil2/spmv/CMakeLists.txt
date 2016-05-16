include_directories(
    common_src
)

# APP_CFLAGS += -ffast-math
# APP_CXXFLAGS += -ffast-math
# APP_CUDACFLAGS += -Xptxas -dlcm=cg -arch compute_30 --use_fast_math

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ffast-math")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math -Xptxas -dlcm=cg")

add_executable(spmv
    file.cc
    gpu_info.cc
    main.cu
    common_src/mmio.c
    common_src/convert_dataset.c
    ../common/src/parboil_cuda.c
)
set_target_properties(spmv PROPERTIES LINKER_LANGUAGE CUDA)
