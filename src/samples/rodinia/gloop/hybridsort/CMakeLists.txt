add_executable(hybridsort-gloop
    main.cu
    bucketsort.cu
    mergesort.cu
)
set_target_properties(hybridsort-gloop PROPERTIES LINKER_LANGUAGE CUDA)
target_link_libraries(hybridsort-gloop gloop ${GLOOP_CUDA_RUNTIME_LIBS})
