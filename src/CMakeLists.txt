# include subdirectory

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++14 -stdlib=libc++")
set(GLOOP_CUDA_RUNTIME_LIBS dl;rt;c++;m;cudart_static;${CMAKE_THREAD_LIBS_INIT})

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
