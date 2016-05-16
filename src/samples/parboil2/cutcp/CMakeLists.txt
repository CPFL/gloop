add_executable(cutcp
    main.c
    readatom.c
    output.c
    excl.c
    cutcpu.c
    cutoff6overlap.cu
    ../common/src/parboil_cuda.c
)
set_target_properties(cutcp PROPERTIES LINKER_LANGUAGE CUDA)
