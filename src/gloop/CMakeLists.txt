ExternalProject_Get_Property(LIBUV_LIB_EP SOURCE_DIR)
include_directories("${SOURCE_DIR}/include")
include_directories(".")

# set(CUDA_RUNTIME_API_HEADER)
# foreach(DIR ${CUDA_INCLUDE_DIRS} OK)
#     if(EXISTS "${DIR}/cuda_runtime_api.h")
#         set(CUDA_RUNTIME_API_HEADER "${DIR}/cuda_runtime_api.h")
#         break()
#     endif()
# endforeach()
#
# add_custom_command(
#     OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/redirector.h"
#     COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/extract-apis.py redirector-header ${CUDA_RUNTIME_API_HEADER} > ${CMAKE_CURRENT_BINARY_DIR}/redirector.h
#     IMPLICIT_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/extract-apis.py
#     WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
# )
#
# add_custom_command(
#     OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/redirector.cc"
#     COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/extract-apis.py redirector-implementation ${CUDA_RUNTIME_API_HEADER} > ${CMAKE_CURRENT_BINARY_DIR}/redirector.cc
#     DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/redirector.h"
#     IMPLICIT_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/extract-apis.py
#     WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
# )
#
# add_custom_target(redirector
#     DEPENDS
#         "${CMAKE_CURRENT_BINARY_DIR}/redirector.cc"
#         "${CMAKE_CURRENT_BINARY_DIR}/redirector.h"
# )

add_library(gloop STATIC
    # gloop_host.cc
    # gloop_device.cu
    # ${CMAKE_CURRENT_BINARY_DIR}/redirector.cc
    host_loop.cu
    device_loop.cu
    monitor_session.cc
)
target_link_libraries(gloop
    dl
    uv
    rt
    boost_system
    boost_thread
    boost_date_time
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
# add_dependencies(gloop uv redirector)
