add_executable(memcopy_gloop
    fs.cu
    cp.cu
    $<TARGET_OBJECTS:backward_object>
)
target_link_libraries(memcopy_gloop gloop ${GLOOP_CUDA_RUNTIME_LIBS})
set_target_properties(memcopy_gloop PROPERTIES LINKER_LANGUAGE CUDA)
