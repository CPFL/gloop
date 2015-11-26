cmake_minimum_required(VERSION 2.8)
project(gloop)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

find_package(CUDA 7.5 REQUIRED)
find_package(Threads)
include(CUDAMacros)

add_subdirectory(third_party)
add_subdirectory(src)
