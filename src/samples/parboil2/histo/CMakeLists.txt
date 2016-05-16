add_executable(histo-gloop
    main.cu
    histo_final.cu
    histo_intermediates.cu
    histo_main.cu
    histo_prescan.cu
    util.cu
    ../common/src/parboil_cuda.c
)
target_link_libraries(histo-gloop gloop ${GLOOP_CUDA_RUNTIME_LIBS})
set_target_properties(histo-gloop PROPERTIES LINKER_LANGUAGE CUDA)
