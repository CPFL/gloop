# include subdirectory

list(APPEND CUDA_NVCC_FLAGS
    -arch=sm_20
    -rdc=true
    --generate-code code=sm_20,arch=compute_20
    # --generate-code code=sm_35,arch=compute_35
    -Xptxas -v
    -Xcompiler -fPIC
    -maxrregcount 32
    --expt-extended-lambda
    -std=c++11
    -w
    --keep
)
# https://github.com/bchretien/cmake-cuda-static
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_SEPARABLE_COMPILATION OFF)
set(BUILD_SHARED_LIBS OFF)

if (CMAKE_BUILD_TYPE MATCHES Release)
    add_definitions(
        -DNDEBUG
        -DDEBUG_NOINLINE=
        -DRELEASE=1
        -DMALLOC_STATS
    )
endif()

if (CMAKE_BUILD_TYPE MATCHES Debug)
    add_definitions(
        -DDEBUG_NOINLINE=__noinline__
        -DMALLOC_STATS
    )
endif()

include_directories(".")

add_subdirectory(gpufs)
add_subdirectory(gloop)
add_subdirectory(samples)
