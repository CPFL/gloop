add_executable(tpacf-gloop
    args.cc
    main.cu
    model_io.cc
    ../common/src/parboil_cuda.c
)
target_link_libraries(tpacf-gloop gloop ${GLOOP_CUDA_RUNTIME_LIBS})
set_target_properties(tpacf-gloop PROPERTIES LINKER_LANGUAGE CUDA)
