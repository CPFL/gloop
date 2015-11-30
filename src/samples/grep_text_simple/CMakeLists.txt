include_directories(../..)
cuda_add_executable_dc(grep_text
    main.cu
    DEPENDS gloop uv
)
target_link_libraries(grep_text uv gloop ${CMAKE_THREAD_LIBS_INIT})
