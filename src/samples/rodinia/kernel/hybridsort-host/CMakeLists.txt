add_executable(hybridsort-host-kernel
    main.cu
    bucketsort.cu
    bucketsort_kernel.cu
    bucketcount.cu
    bucketprefixoffset.cu
    mergepack.cu
    mergesort.cu
    mergesort_first_pass.cu
    mergesort_statics.cu
    histogram1024.cu
)
set_target_properties(hybridsort-host-kernel PROPERTIES LINKER_LANGUAGE CUDA)
target_link_libraries(hybridsort-host-kernel gloop ${GLOOP_CUDA_RUNTIME_LIBS})
