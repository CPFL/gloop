if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

cmake_minimum_required(VERSION 2.8)
project(gloop C CXX CUDA)

find_package(Threads)

macro(gloop_cuda_include_directories)
  foreach(dir ${ARGN})
    if(IS_ABSOLUTE "${dir}")
      set(resolved_dir "${dir}")
    else()
      set(resolved_dir "${CMAKE_CURRENT_SOURCE_DIR}/${dir}")
    endif()
    list(APPEND CUDA_NVCC_INCLUDE_ARGS_USER -I${resolved_dir})
  endforeach()
endmacro()

include_directories("${PROJECT_BINARY_DIR}/include")

add_subdirectory(third_party)
add_subdirectory(src)
add_subdirectory(test)
# add_subdirectory(tools)
