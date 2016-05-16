add_executable(mri-gridding
    CPU_kernels.c
    CUDA_interface.cu
    scanLargeArray.cu
    sort.cu
    main.cu
    ../common/src/parboil_cuda.c
)
set_target_properties(mri-gridding PROPERTIES LINKER_LANGUAGE CUDA)
