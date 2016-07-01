#ifndef __BUCKETSORT
#define __BUCKETSORT

#include <gloop/gloop.h>

#define LOG_DIVISIONS 10
#define DIVISIONS (1 << LOG_DIVISIONS)

#define BUCKET_WARP_LOG_SIZE 5
#define BUCKET_WARP_N 1
#ifdef BUCKET_WG_SIZE_1
#define BUCKET_THREAD_N BUCKET_WG_SIZE_1
#else
#define BUCKET_THREAD_N (BUCKET_WARP_N << BUCKET_WARP_LOG_SIZE)
#endif
#define BUCKET_BLOCK_MEMORY (DIVISIONS * BUCKET_WARP_N)
#define BUCKET_BAND 128

void init_bucketsort(int listsize);
void finish_bucketsort();
void bucketSort(gloop::HostLoop&, gloop::HostContext&, float* d_input, float* d_output, int listsize, int* sizes, int* nullElements, float minimum, float maximum, unsigned int* origOffsets);
void bucketcountGPU(gloop::HostLoop& hostLoop, gloop::HostContext& hostContext, dim3 blocks, dim3 threads, float* input, int* indice, unsigned int* d_prefixoffsets, int size);
void bucketprefixoffsetGPU(gloop::HostLoop& hostLoop, gloop::HostContext& hostContext, dim3 blocks, dim3 threads, unsigned int* d_prefixoffsets, unsigned int* d_offsets, int aBlocks);
void bucketsortGPU(gloop::HostLoop& hostLoop, gloop::HostContext& hostContext, dim3 blocks, dim3 threads, float* input, int* indice, float* output, int size, unsigned int* d_prefixoffsets, unsigned int* l_offsets);

extern texture<float, 1, cudaReadModeElementType> texPivot;

#endif
