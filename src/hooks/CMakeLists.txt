cuda_add_library(hooks SHARED
    main_loop.cc
    redirector.cc
)
link_libraries(hooks dl)
