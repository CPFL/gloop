add_executable(sad
    file.c
    image.c
    largerBlocks.cu
    main.cu
    sad4.cu
    ../common/src/parboil_cuda.c
)
set_target_properties(sad PROPERTIES LINKER_LANGUAGE CUDA)
