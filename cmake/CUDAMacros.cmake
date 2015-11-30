# From https://github.com/bchretien/cmake-cuda-static/

set(CUDA_C_OR_CXX CXX)


macro(CUDA_GET_SOURCES_AND_OPTIONS2 _sources _dependencies _options _linker_options)
    set(${_sources})
    set(${_dependencies})
    set(${_options})
    set(${_linker_options})
    set(_mode "SOURCES")
    set(_found_options FALSE )
    foreach(arg ${ARGN})
        if("x${arg}" STREQUAL "xDEPENDS")
            set(_mode "DEPENDS")
        elseif("x${arg}" STREQUAL "xOPTIONS")
            set(_mode "OPTIONS")
        elseif("x${arg}" STREQUAL "xLINKER_OPTIONS")
            set(_mode "LINKER_OPTIONS")
        else()
            if("x${_mode}" STREQUAL "xSOURCES")
                list(APPEND ${_sources} ${arg})
            elseif("x${_mode}" STREQUAL "xDEPENDS")
                list(APPEND ${_dependencies} ${arg})
            elseif("x${_mode}" STREQUAL "xOPTIONS")
                list(APPEND ${_options} ${arg})
            elseif("x${_mode}" STREQUAL "xLINKER_OPTIONS")
                list(APPEND ${_linker_options} ${arg})
            endif()
        endif()
    endforeach()
endmacro()


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
MACRO(CUDA_ADD_EXECUTABLE_DC cuda_target)
    CMAKE_POLICY(SET CMP0026 OLD)
    CMAKE_POLICY(SET CMP0039 OLD)

    CUDA_ADD_CUDA_INCLUDE_ONCE()

    CUDA_GET_SOURCES_AND_OPTIONS2(_sources _dependencies _options _linker_options ${ARGN})

    set(CUDA_NVCC_INCLUDE_ARGS ${CUDA_NVCC_INCLUDE_ARGS_USER} "-I${CUDA_INCLUDE_DIRS}")
    get_directory_property(CUDA_NVCC_INCLUDE_DIRECTORIES INCLUDE_DIRECTORIES)
    list(REMOVE_DUPLICATES CUDA_NVCC_INCLUDE_DIRECTORIES)
    if(CUDA_NVCC_INCLUDE_DIRECTORIES)
        foreach(dir ${CUDA_NVCC_INCLUDE_DIRECTORIES})
            list(APPEND CUDA_NVCC_INCLUDE_ARGS -I${dir})
        endforeach()
    endif()

    # Separate the sources from the options
    # CUDA_GET_SOURCES_AND_OPTIONS(_sources _cmake_options _options ${ARGN})
    # Create custom commands and targets for each file.
    # CUDA_WRAP_SRCS( ${cuda_target} OBJ _generated_files ${_sources} OPTIONS ${_options} )

    CUDA_WRAP_SRCS(${cuda_target} OBJ _generated_files ${_sources} OPTIONS ${_options})
    ADD_CUSTOM_TARGET(${cuda_target} ALL DEPENDS ${_dependencies} ${_generated_files} ${_sources})

    # List of dependencies (static libraries)
    SET(dep_paths "")
    FOREACH(DEP ${_dependencies})
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
        TARGET ${cuda_target}
        PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E echo "Linking static device library to executable."
        COMMAND ${CUDA_NVCC_EXECUTABLE} ${CUDA_NVCC_INCLUDE_ARGS} ${CUDA_NVCC_FLAGS} ${_options} ${nvcc_flags} -dlink ${dep_paths} ${_generated_files}
        COMMAND ${CUDA_NVCC_EXECUTABLE} ${CUDA_NVCC_INCLUDE_ARGS} ${CUDA_NVCC_FLAGS} ${_options} ${dep_paths} ${_generated_files} ${_linker_options} -o "${cuda_target}"
        )

    SET_TARGET_PROPERTIES(${cuda_target}
        PROPERTIES
        LINKER_LANGUAGE ${CUDA_C_OR_CXX})
ENDMACRO(CUDA_ADD_EXECUTABLE_DC)

