add_executable(tpacf-non-shared
    args.cc
    main.cu
    model_io.cc
    ../common/src/parboil_cuda.c
)
set_target_properties(tpacf-non-shared PROPERTIES LINKER_LANGUAGE CUDA)
target_link_libraries(tpacf-non-shared ${GLOOP_CUDA_RUNTIME_LIBS})
