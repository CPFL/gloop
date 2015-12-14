add_library(gpufs STATIC
    cpu_ipc.cu
    fs_calls.cu
    fs_initializer.cu
    fs_structures.cu
    hash_table.cu
    mallocfree.cu
    preclose_table.cu
    radix_tree.cu
    swapper.cu
    timer.cpp
    fs_debug.cu
    async_ipc.cu
    generic_ringbuf.cu
)
