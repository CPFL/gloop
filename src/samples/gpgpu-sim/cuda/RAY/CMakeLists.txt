include_directories(.)
add_executable(RAY
    rayTracing.cu
    EasyBMP.cpp
    makebmp.cpp
)
target_link_libraries(RAY ${GLOOP_CUDA_RUNTIME_LIBS})
set_target_properties(RAY PROPERTIES LINKER_LANGUAGE CUDA)
