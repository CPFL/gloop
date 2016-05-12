# add_executable(matmul
#     matrixMul.cu
#     $<TARGET_OBJECTS:backward_object>
# )
# target_link_libraries(matmul gloop ${GLOOP_CUDA_RUNTIME_LIBS})
# set_target_properties(matmul PROPERTIES LINKER_LANGUAGE CUDA)

# add_executable(matmul_check
#     matrixMulCheck.cu
# )
# target_link_libraries(matmul_check gloop ${GLOOP_CUDA_RUNTIME_LIBS})
# set_target_properties(matmul_check PROPERTIES LINKER_LANGUAGE CUDA)
