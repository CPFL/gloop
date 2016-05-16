add_executable(histo
    main.cu
    histo_final.cu
    histo_intermediates.cu
    histo_main.cu
    histo_prescan.cu
    util.cu
    ../common/src/parboil_cuda.c
)
set_target_properties(histo PROPERTIES LINKER_LANGUAGE CUDA)
