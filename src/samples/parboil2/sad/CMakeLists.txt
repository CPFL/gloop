add_executable(sad-gloop
    file.c
    image.c
    largerBlocks.cu
    main.cu
    sad4.cu
    ../common/src/parboil_cuda.c
)
target_link_libraries(sad-gloop gloop ${GLOOP_CUDA_RUNTIME_LIBS})
set_target_properties(sad-gloop PROPERTIES LINKER_LANGUAGE CUDA)
