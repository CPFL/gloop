list(APPEND CMAKE_MODULE_PATH "/usr/local/share/llvm/cmake")

include(AddLLVM)
include(HandleLLVMOptions)
include(LLVMConfig)

llvm_process_sources(srcs
    rewrite_lambda.cc
)

include_directories("/usr/local/include")
link_directories("/usr/local/lib")
add_library(rewrite_lambda MODULE ${srcs})

target_link_libraries(rewrite_lambda
    clangFrontend
    clangRewrite
    clangAST
    clangLex
    clangBasic
)
