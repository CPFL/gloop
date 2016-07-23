# include subdirectory

# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++14 -stdlib=libc++")
# set(GLOOP_CUDA_RUNTIME_LIBS cudart_static;dl;rt;c++;m;${CMAKE_THREAD_LIBS_INIT})

# set(CMAKE_CXX_FLAGS "-g3 -std=c++11")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G -rdc=true --expt-extended-lambda -std=c++11 -w --keep")
set(CMAKE_CXX_FLAGS "-std=c++11 -Wall -Wno-unused-parameter -g3")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -maxrregcount 32 --generate-code code=sm_30,arch=compute_30 -rdc=true --expt-extended-lambda -std=c++11 -w --keep")
#    --generate-code code=sm_30,arch=compute_30\
set(CMAKE_CUDA_FLAGS "\
    ${CMAKE_CUDA_FLAGS}\
    -maxrregcount 32\
    -rdc=true\
    --generate-code code=sm_35,arch=compute_35\
    --expt-extended-lambda\
    -std=c++11\
    -Xptxas -v,--warn-on-spills,--warn-on-local-memory-usage\
    -Xcompiler -Wall,-Wno-unused-parameter,-Wextra\
    -g\
    -w\
    --keep\
")
#  set(CMAKE_CUDA_FLAGS "\
#      ${CMAKE_CUDA_FLAGS}\
#      -maxrregcount 32\
#      --generate-code code=sm_30,arch=compute_30\
#      -rdc=true\
#      --expt-extended-lambda\
#      -std=c++11\
#      -Xptxas -v,--warn-on-spills,--warn-on-local-memory-usage\
#      -Xcompiler -Wall,-Wno-unused-parameter,-Wextra\
#      -w\
#      --keep\
#      -G\
#  ")
# -Xptxas --warn-on-spills,--warn-on-local-memory-usage\

set(GLOOP_CUDA_RUNTIME_LIBS cuda cudart;dl;rt;m;${CMAKE_THREAD_LIBS_INIT})
link_directories(/usr/local/cuda-7.5/lib64)

add_definitions(
#     -D_FORCE_INLINES
    -D_HAVE_STRING_ARCH_mempcpy
#     -D_MWAITXINTRIN_H_INCLUDED
)

if (CMAKE_BUILD_TYPE MATCHES Release)
    add_definitions(
        -DNDEBUG
        -DDEBUG_NOINLINE=
        -DRELEASE=1
        -DMALLOC_STATS
    )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O2")
endif()

if (CMAKE_BUILD_TYPE MATCHES Debug)
    add_definitions(
        -DDEBUG_NOINLINE=__noinline__
        -DMALLOC_STATS
    )
endif()

include_directories(".")
include_directories("${CMAKE_CURRENT_BINARY_DIR}/gloop")
include_directories("${CUDA_RUNTIME_INCLUDE_DIR}")

# add_subdirectory(cmalloc)
# add_subdirectory(gpufs)
add_subdirectory(gloop)
add_subdirectory(samples)
