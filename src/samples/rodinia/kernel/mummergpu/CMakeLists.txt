find_package(OpenMP REQUIRED)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
# Short term solution.
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler ${OpenMP_CXX_FLAGS}")

include_directories(src)
add_executable(mummergpu-kernel
    src/mummergpu.cu
    src/mummergpu_gold.cpp
    src/mummergpu_main.cpp
    src/PoolMalloc.cpp
    src/suffix-tree.cpp
)
set_target_properties(mummergpu-kernel PROPERTIES LINKER_LANGUAGE CUDA)
target_link_libraries(mummergpu-kernel ${GLOOP_CUDA_RUNTIME_LIBS})
