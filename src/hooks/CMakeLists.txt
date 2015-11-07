cuda_add_library(hooks SHARED
    main-loop.cc
    redirector.cc
)
link_libraries(hooks dl)
