include_directories(../..)
add_executable(grep_text
    main.cu
)
target_link_libraries(grep_text uv ${GLOOP_CUDA_RUNTIME_LIBS} gloop)
