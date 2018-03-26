add_executable(dma
    main.cu
)
target_link_libraries(dma ${GLOOP_CUDA_RUNTIME_LIBS})
set_target_properties(dma PROPERTIES LINKER_LANGUAGE CUDA)
