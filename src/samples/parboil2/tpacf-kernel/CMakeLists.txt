add_executable(tpacf-kernel
    args.cc
    main.cu
    model_io.cc
    ../common/src/parboil_cuda.c
)
set_target_properties(tpacf-kernel PROPERTIES LINKER_LANGUAGE CUDA)
target_link_libraries(tpacf-kernel ${GLOOP_CUDA_RUNTIME_LIBS})
