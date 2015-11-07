cuda_add_library(hooks SHARED
    library.cu
    redirector.cu
)
link_libraries(hooks dl)
