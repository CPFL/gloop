add_executable(hybridsort-gloop
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
set_target_properties(hybridsort-gloop PROPERTIES LINKER_LANGUAGE CUDA)
target_link_libraries(hybridsort-gloop gloop ${GLOOP_CUDA_RUNTIME_LIBS})
