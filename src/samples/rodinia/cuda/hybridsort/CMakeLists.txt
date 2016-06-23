add_executable(hybridsort
    main.cu
    bucketsort.cu
    mergesort.cu
)
set_target_properties(hybridsort PROPERTIES LINKER_LANGUAGE CUDA)
target_link_libraries(hybridsort ${GLOOP_CUDA_RUNTIME_LIBS})
