add_executable(hybridsort-host
    main.cu
    bucketsort.cu
    mergesort.cu
)
set_target_properties(hybridsort-host PROPERTIES LINKER_LANGUAGE CUDA)
target_link_libraries(hybridsort-host ${GLOOP_CUDA_RUNTIME_LIBS})
