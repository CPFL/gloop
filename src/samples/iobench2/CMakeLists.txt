add_executable(iobench2
    fs.cu
    cp.cu
)
target_link_libraries(iobench2 gloop ${GLOOP_CUDA_RUNTIME_LIBS})
set_target_properties(iobench2 PROPERTIES LINKER_LANGUAGE CUDA)
