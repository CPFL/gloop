add_executable(tpacf
    args.cc
    main.cu
    model_io.cc
    ../common/src/parboil_cuda.c
)
set_target_properties(tpacf PROPERTIES LINKER_LANGUAGE CUDA)
