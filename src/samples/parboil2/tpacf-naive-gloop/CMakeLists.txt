add_executable(tpacf-naive-gloop
    args.cc
    main.cu
    model_io.cc
    ../common/src/parboil_cuda.c
)
target_link_libraries(tpacf-naive-gloop gloop ${GLOOP_CUDA_RUNTIME_LIBS})
set_target_properties(tpacf-naive-gloop PROPERTIES LINKER_LANGUAGE CUDA)
