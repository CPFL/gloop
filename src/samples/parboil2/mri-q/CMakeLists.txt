add_executable(mri-q
    file.cc
    main.cu
    ../common/src/parboil_cuda.c
)
set_target_properties(mri-q PROPERTIES LINKER_LANGUAGE CUDA)
