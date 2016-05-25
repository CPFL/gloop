add_executable(lbm-gloop
    lbm.cu
    main.cu
    ../common/src/parboil_cuda.c
)
target_link_libraries(lbm-gloop gloop ${GLOOP_CUDA_RUNTIME_LIBS})
set_target_properties(lbm-gloop PROPERTIES LINKER_LANGUAGE CUDA)
