add_executable(mri-gridding-gloop
    CPU_kernels.c
    CUDA_interface.cu
    scanLargeArray.cu
    sort.cu
    main.cu
    ../common/src/parboil_cuda.c
)
target_link_libraries(mri-gridding-gloop gloop ${GLOOP_CUDA_RUNTIME_LIBS})
set_target_properties(mri-gridding-gloop PROPERTIES LINKER_LANGUAGE CUDA)
