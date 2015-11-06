add_library(hooks SHARED
    library.cc
)
link_libraries(hooks dl)
