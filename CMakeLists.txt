cmake_minimum_required(VERSION 2.8)
project(node.cu)

find_package(CUDA REQUIRED)
# set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
# message(${CMAKE_MODULE_PATH})

# find_package(Threads)

# setup google-test target
add_subdirectory(src)
