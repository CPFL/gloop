cuda_add_library(hooks SHARED
    library.cc
    redirector.cc
)
link_libraries(hooks dl)
