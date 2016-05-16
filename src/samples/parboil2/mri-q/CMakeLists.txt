add_executable(mri-q-gloop
    file.cc
    main.cu
    ../common/src/parboil_cuda.c
)
target_link_libraries(mri-q-gloop gloop ${GLOOP_CUDA_RUNTIME_LIBS})
set_target_properties(mri-q-gloop PROPERTIES LINKER_LANGUAGE CUDA)
