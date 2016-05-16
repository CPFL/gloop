add_executable(lbm
    lbm.cu
    main.cc
    ../common/src/parboil_cuda.c
)
set_target_properties(lbm PROPERTIES LINKER_LANGUAGE CUDA)
