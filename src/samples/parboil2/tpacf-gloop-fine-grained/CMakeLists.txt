add_executable(tpacf-gloop-fine-grained
    args.cc
    main.cu
    model_io.cc
    ../common/src/parboil_cuda.c
)
target_link_libraries(tpacf-gloop-fine-grained gloop ${GLOOP_CUDA_RUNTIME_LIBS})
set_target_properties(tpacf-gloop-fine-grained PROPERTIES LINKER_LANGUAGE CUDA)
