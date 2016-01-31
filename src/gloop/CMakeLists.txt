cmake_policy(SET CMP0055 NEW)

ExternalProject_Get_Property(LIBUV_LIB_EP SOURCE_DIR)
include_directories("${SOURCE_DIR}/include")
include_directories(".")
include_directories("../")
include_directories("../gpufs/gpufs_con_lib.user/")
include_directories("${CMAKE_CURRENT_BINARY_DIR}")

set(CUDA_RUNTIME_API_HEADER)
if(EXISTS "${CUDA_RUNTIME_INCLUDE_DIR}/cuda_runtime_api.h")
    set(CUDA_RUNTIME_API_HEADER "${CUDA_RUNTIME_INCLUDE_DIR}/cuda_runtime_api.h")
endif()

add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/redirector.cuh"
    COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/extract-apis.py redirector-header ${CUDA_RUNTIME_API_HEADER} > ${CMAKE_CURRENT_BINARY_DIR}/redirector.cuh
    IMPLICIT_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/extract-apis.py
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/redirector.cu"
    COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/extract-apis.py redirector-implementation ${CUDA_RUNTIME_API_HEADER} > ${CMAKE_CURRENT_BINARY_DIR}/redirector.cu
    DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/redirector.cuh"
    IMPLICIT_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/extract-apis.py
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

add_custom_target(redirector
    DEPENDS
        "${CMAKE_CURRENT_BINARY_DIR}/redirector.cu"
        "${CMAKE_CURRENT_BINARY_DIR}/redirector.cuh"
)

add_library(gloop STATIC
    # gloop_device.cu
    ${CMAKE_CURRENT_BINARY_DIR}/redirector.cu
    host_context.cu
    host_loop.cu
    host_memory.cu
    device_loop.cu
    device_memory.cu
    dump_memory.cu
    fs.cu
    io.cu
    ipc.cu
    mapped_memory.cu
    monitor_session.cc
    process.cu
    system_initialize.cu
)
target_link_libraries(gloop
    dl
    uv
    rt
    boost_system
    boost_thread
    boost_date_time
    gipc
)

add_executable(gloop-monitor
    monitor.cc
)
target_link_libraries(gloop-monitor
    gloop
    rt
    ${CMAKE_THREAD_LIBS_INIT}
    boost_system
    boost_thread
    boost_date_time
)
add_dependencies(gloop uv redirector)
