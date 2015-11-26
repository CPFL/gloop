# From https://github.com/bchretien/cmake-cuda-static/

CMAKE_POLICY(SET CMP0026 OLD)
CMAKE_POLICY(SET CMP0039 OLD)
set(CUDA_C_OR_CXX CXX)

##############################################################################
# This helper macro generates an executable that links with a static device 
# library generated with nvcc.
# INPUT:
#   target              - Target name
#   src_files           - List of source files.
#   dependencies        - List of dependencies.
#   options             - Compiler options.
#   linker_options      - Linker options.
##############################################################################
##############################################################################
MACRO(CUDA_ADD_EXECUTABLE_DC target src_files dependencies options linker_options)
  CUDA_ADD_CUDA_INCLUDE_ONCE()

  SET(obj_files "")
  FOREACH(f ${src_files})
    GET_FILENAME_COMPONENT(obj_file ${f} NAME_WE)
    SET(obj_files "${obj_files};${obj_file}.o")
  ENDFOREACH(f ${src_files})

  ADD_CUSTOM_TARGET(${target} ALL DEPENDS ${dependencies})

  # List of dependencies (static libraries)
  SET(dep_paths "")
  FOREACH(DEP ${dependencies})
    GET_PROPERTY(dep_location TARGET ${DEP} PROPERTY LOCATION)
    SET(dep_paths "${dep_paths};${dep_location}")
  ENDFOREACH(DEP)

  # Get the list of definitions from the directory property
  GET_DIRECTORY_PROPERTY(CUDA_NVCC_DEFINITIONS COMPILE_DEFINITIONS)
  IF(CUDA_NVCC_DEFINITIONS)
    FOREACH(_definition ${CUDA_NVCC_DEFINITIONS})
      LIST(APPEND nvcc_flags "-D${_definition}")
    ENDFOREACH()
  ENDIF()

  # Compile and link the static device library with nvcc
  ADD_CUSTOM_COMMAND(
    TARGET ${target}
    PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "Linking static device library to executable."
    COMMAND ${CUDA_NVCC_EXECUTABLE} ${CUDA_NVCC_FLAGS} ${options} ${nvcc_flags} -dlink ${dep_paths} -c "${src_files}"
    COMMAND ${CUDA_NVCC_EXECUTABLE} ${CUDA_NVCC_FLAGS} ${options} ${dep_paths} ${obj_files} ${linker_options} -o "${target}"
    )

  SET_TARGET_PROPERTIES(${target}
    PROPERTIES
    LINKER_LANGUAGE ${CUDA_C_OR_CXX})
ENDMACRO(CUDA_ADD_EXECUTABLE_DC)

