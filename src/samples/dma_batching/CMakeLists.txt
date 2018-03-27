add_executable(dma_batching
    main.cu
)
target_link_libraries(dma_batching ${GLOOP_CUDA_RUNTIME_LIBS})
set_target_properties(dma_batching PROPERTIES LINKER_LANGUAGE CUDA)
