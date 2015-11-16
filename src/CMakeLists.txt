# include subdirectory

# http://stackoverflow.com/questions/29121211/cuda-compilation-issue-with-cmake
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
list(APPEND CUDA_NVCC_FLAGS "-arch=sm_20;-std=c++11;-O2;-DVERBOSE;--expt-extended-lambda")
set(CUDA_PROPAGATE_HOST_FLAGS OFF)

add_subdirectory(gloop)
add_subdirectory(samples)
