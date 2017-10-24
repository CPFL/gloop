cmake_policy(SET CMP0055 NEW)
ExternalProject_Get_Property(LIBUV_LIB_EP SOURCE_DIR)
include_directories("${SOURCE_DIR}/include")
include_directories(".")
include_directories("../")
# include_directories("../gpufs/gpufs_con_lib.user/")
include_directories("${CMAKE_CURRENT_BINARY_DIR}")

option(GLOOP_ENABLE_HIERARCHICAL_SLOT_MEMORY "Enable hierarchical slot memory" ON)
option(GLOOP_ENABLE_ELASTIC_KERNELS "Enable elastic kernels" ON)
option(GLOOP_ENABLE_IO_BOOSTING "Enable IO boosting" ON)
option(GLOOP_ENABLE_CLOCK_CHECK "Enable clock checking" ON)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/generated_config.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/generated_config.h
)

set(CUDA_RUNTIME_API_HEADER)
if(EXISTS "${CUDA_RUNTIME_INCLUDE_DIR}/cuda_runtime_api.h")
    set(CUDA_RUNTIME_API_HEADER "${CUDA_RUNTIME_INCLUDE_DIR}/cuda_runtime_api.h")
endif()

add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/redirector.cuh"
    COMMAND python "${CMAKE_CURRENT_SOURCE_DIR}/extract-apis.py" redirector-header ${CUDA_RUNTIME_API_HEADER} > "${CMAKE_CURRENT_BINARY_DIR}/redirector.cuh"
    IMPLICIT_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/extract-apis.py"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/redirector.cu"
    COMMAND python "${CMAKE_CURRENT_SOURCE_DIR}/extract-apis.py" redirector-implementation "${CUDA_RUNTIME_API_HEADER}" > "${CMAKE_CURRENT_BINARY_DIR}/redirector.cu"
    DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/redirector.cuh"
    IMPLICIT_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/extract-apis.py"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

add_custom_target(redirector
    DEPENDS
        "${CMAKE_CURRENT_BINARY_DIR}/redirector.cu"
        "${CMAKE_CURRENT_BINARY_DIR}/redirector.cuh"
)

add_library(gloop STATIC
    # gloop_device.cu
    "${CMAKE_CURRENT_BINARY_DIR}/redirector.cu"
    copy_work.cu
    copy_work_pool.cu
    copy_worker.cu
    host_context.cu
    host_loop.cu
    host_memory.cu
    device_loop.cu
    device_memory.cu
    dump_memory.cu
    io.cu
    mapped_memory.cu
    process.cu
    system_initialize.cu
    data_log.cu
    monitor_utility.cu
)
target_link_libraries(gloop
    dl
    rt
    boost_system
    boost_thread
    boost_date_time
)
add_dependencies(gloop redirector)

find_program(GRPC_CPP_PLUGIN_PATH
    NAMES "grpc_cpp_plugin"
)
message("GRPC:${GRPC_CPP_PLUGIN_PATH}")
if("${GRPC_CPP_PLUGIN_PATH}" STREQUAL "GRPC_CPP_PLUGIN_PATH-NOTFOUND")
    message(FATAL_ERROR "GRPC: grpc_cpp_plugin is not found.")
endif()

add_custom_command(
    OUTPUT
        "${CMAKE_CURRENT_BINARY_DIR}/monitor_service.pb.cc"
        "${CMAKE_CURRENT_BINARY_DIR}/monitor_service.pb.h"
    COMMAND
        "${PROTOC}" "${CMAKE_CURRENT_SOURCE_DIR}/monitor_service.proto" --proto_path="${CMAKE_CURRENT_SOURCE_DIR}" --cpp_out="${CMAKE_CURRENT_BINARY_DIR}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/monitor_service.proto" protobuf
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

add_custom_command(
    OUTPUT
        "${CMAKE_CURRENT_BINARY_DIR}/monitor_service.grpc.pb.cc"
        "${CMAKE_CURRENT_BINARY_DIR}/monitor_service.grpc.pb.h"
    COMMAND
        "${PROTOC}" "${CMAKE_CURRENT_SOURCE_DIR}/monitor_service.proto" --proto_path="${CMAKE_CURRENT_SOURCE_DIR}" --grpc_out="${CMAKE_CURRENT_BINARY_DIR}" --plugin=protoc-gen-grpc="${GRPC_CPP_PLUGIN_PATH}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/monitor_service.proto" protobuf
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

add_custom_target(monitor-service
    DEPENDS
        "${CMAKE_CURRENT_BINARY_DIR}/monitor_service.pb.cc"
        "${CMAKE_CURRENT_BINARY_DIR}/monitor_service.pb.h"
        "${CMAKE_CURRENT_BINARY_DIR}/monitor_service.grpc.pb.cc"
        "${CMAKE_CURRENT_BINARY_DIR}/monitor_service.grpc.pb.h"
)

add_executable(gloop-monitor
    monitor.cc
    monitor_main.cc
    monitor_server.cc
    monitor_session.cc
    monitor_utilization_accounting.cc
    "${CMAKE_CURRENT_BINARY_DIR}/monitor_service.pb.cc"
    "${CMAKE_CURRENT_BINARY_DIR}/monitor_service.grpc.pb.cc"
    data_log.cc
    monitor_utility.cc
)
target_link_libraries(gloop-monitor
    grpc++_unsecure
    grpc
    gloop
    rt
    ${CMAKE_THREAD_LIBS_INIT}
    boost_system
    boost_thread
    boost_date_time
    protobuf
)
add_dependencies(gloop-monitor monitor-service)


file(GLOB_RECURSE ALL_CXX_SOURCE_FILES
    *.[chi]pp *.[chi]xx *.cc *.hh *.ii *.[CHI] *.cu *.cuh
)

add_custom_target(
    format
    COMMAND /usr/bin/clang-format
    -i
    ${ALL_CXX_SOURCE_FILES}
)
