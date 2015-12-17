list(APPEND CMAKE_MODULE_PATH "/usr/local/share/llvm/cmake")

include(AddLLVM)
include(HandleLLVMOptions)
include(LLVMConfig)

llvm_process_sources(srcs
    rewrite_lambda.cc
)

include_directories("/usr/local/include")
link_directories("/usr/local/lib")
add_llvm_loadable_module(rewrite_lambda ${srcs})

target_link_libraries(rewrite_lambda
    clang
    clangAST
    clangBasic
    clangFrontend
    clangLex
    clangRewrite
    LLVMSupport
)
