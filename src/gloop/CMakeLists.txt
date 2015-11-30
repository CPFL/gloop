ExternalProject_Get_Property(LIBUV_LIB_EP SOURCE_DIR)
include_directories("${SOURCE_DIR}/include")
include_directories("${CMAKE_CURRENT_BINARY_DIR}")
include_directories(".")

set(CUDA_RUNTIME_API_HEADER)
foreach(DIR ${CUDA_INCLUDE_DIRS} OK)
    if(EXISTS "${DIR}/cuda_runtime_api.h")
        set(CUDA_RUNTIME_API_HEADER "${DIR}/cuda_runtime_api.h")
        break()
    endif()
endforeach()

if(${CUDA_RUNTIME_API_HEADER} EQUAL "")
    message(FATAL_ERROR "CUDA Runtime API header does not exist.")
else()
    message("-- CUDA Runtime API header: ${CUDA_RUNTIME_API_HEADER}")
endif()

add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/redirector.h"
    COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/extract-apis.py redirector-header ${CUDA_RUNTIME_API_HEADER} > ${CMAKE_CURRENT_BINARY_DIR}/redirector.h
    IMPLICIT_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/extract-apis.py
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/redirector.cc"
    COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/extract-apis.py redirector-implementation ${CUDA_RUNTIME_API_HEADER} > ${CMAKE_CURRENT_BINARY_DIR}/redirector.cc
    DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/redirector.h"
    IMPLICIT_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/extract-apis.py
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

add_custom_target(redirector
    DEPENDS
        "${CMAKE_CURRENT_BINARY_DIR}/redirector.cc"
        "${CMAKE_CURRENT_BINARY_DIR}/redirector.h"
)

cuda_add_library(gloop STATIC
    # gloop_host.cc
    # gloop_device.cu
    # host_loop.cc
    # ${CMAKE_CURRENT_BINARY_DIR}/redirector.cc
    fs.cu
    device_loop.cu
)
add_dependencies(gloop uv redirector)
target_link_libraries(gloop dl uv)
