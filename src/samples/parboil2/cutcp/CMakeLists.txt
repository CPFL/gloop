add_executable(cutcp-gloop
    main.c
    readatom.c
    output.c
    excl.c
    cutcpu.c
    cutoff6overlap.cu
    ../common/src/parboil_cuda.c
)
target_link_libraries(cutcp-gloop gloop ${GLOOP_CUDA_RUNTIME_LIBS})
set_target_properties(cutcp-gloop PROPERTIES LINKER_LANGUAGE CUDA)
