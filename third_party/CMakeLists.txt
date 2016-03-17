# Based on https://github.com/iauns/cpm-libuv/
include(ExternalProject)

# libuv
set(EP_TARGET_NAME LIBUV_LIB_EP)
set(LIBUV_LOCATION ${PROJECT_BINARY_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}uv${CMAKE_STATIC_LIBRARY_SUFFIX})
ExternalProject_Add(${EP_TARGET_NAME}
    URL ${CMAKE_CURRENT_SOURCE_DIR}/libuv-v1.7.5.tar.gz
    CONFIGURE_COMMAND sh <SOURCE_DIR>/autogen.sh && ./configure --prefix=<INSTALL_DIR>
    PATCH_COMMAND ""
    INSTALL_DIR ${PROJECT_BINARY_DIR}
    BUILD_COMMAND "make"
    BUILD_IN_SOURCE 1
    BUILD_BYPRODUCTS ${LIBUV_LOCATION}
)
add_library(uv STATIC IMPORTED GLOBAL)
set_target_properties(uv
    PROPERTIES
    IMPORTED_LOCATION ${LIBUV_LOCATION}
)
add_dependencies(uv ${EP_TARGET_NAME})

# protobuf
set(EP_TARGET_NAME PROTOBUF_EP)
set(PROTOBUF_LOCATION ${PROJECT_BINARY_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}protobuf${CMAKE_STATIC_LIBRARY_SUFFIX})
ExternalProject_Add(${EP_TARGET_NAME}
    URL ${CMAKE_CURRENT_SOURCE_DIR}/protobuf-cpp-3.0.0-beta-2.tar.gz
    CONFIGURE_COMMAND sh <SOURCE_DIR>/autogen.sh && ./configure --prefix=<INSTALL_DIR>
    PATCH_COMMAND ""
    INSTALL_DIR ${PROJECT_BINARY_DIR}
    BUILD_COMMAND "make"
    BUILD_IN_SOURCE 1
    BUILD_BYPRODUCTS ${PROTOBUF_LOCATION}
)
add_library(protobuf STATIC IMPORTED GLOBAL)
set_target_properties(protobuf
    PROPERTIES
    IMPORTED_LOCATION ${PROTOBUF_LOCATION}
)
add_dependencies(protobuf ${EP_TARGET_NAME})
set(PROTOC ${PROJECT_BINARY_DIR}/bin/protoc)

add_subdirectory(backward-cpp)
add_subdirectory(echo_server)
add_subdirectory(uv_echo)
