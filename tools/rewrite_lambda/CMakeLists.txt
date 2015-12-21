list(APPEND CMAKE_MODULE_PATH "/usr/local/share/llvm/cmake")

include(AddLLVM)
include(HandleLLVMOptions)
include(LLVMConfig)

include_directories("/usr/local/include")
link_directories("/usr/local/lib")
add_llvm_loadable_module(rewrite_lambda rewrite_lambda.cc)
# Should not perform target_link_libraries onto this with LLVM libs.
