include_directories(.)
add_executable(RAY-gloop
    rayTracing.cu
    EasyBMP.cpp
    makebmp.cpp
)
target_link_libraries(RAY-gloop gloop ${GLOOP_CUDA_RUNTIME_LIBS})
set_target_properties(RAY-gloop PROPERTIES LINKER_LANGUAGE CUDA)
