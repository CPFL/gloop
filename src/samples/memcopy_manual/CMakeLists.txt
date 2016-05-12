add_executable(memcopy_manual
    fs.cu
    cp.cu
)
target_link_libraries(memcopy_manual gloop ${GLOOP_CUDA_RUNTIME_LIBS})
set_target_properties(memcopy_manual PROPERTIES LINKER_LANGUAGE CUDA)
