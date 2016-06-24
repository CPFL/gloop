#ifndef _BUCKETSORT_KERNEL_H_
#define _BUCKETSORT_KERNEL_H_

#include <stdio.h>

static __global__ void bucketprefixoffset(unsigned int* d_prefixoffsets, unsigned int* d_offsets, int blocks)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int size = blocks * BUCKET_BLOCK_MEMORY;
    int sum = 0;

    for (int i = tid; i < size; i += DIVISIONS) {
        int x = d_prefixoffsets[i];
        d_prefixoffsets[i] = sum;
        sum += x;
    }

    d_offsets[tid] = sum;
}

static __global__ void
bucketsort(float* input, int* indice, float* output, int size, unsigned int* d_prefixoffsets,
    unsigned int* l_offsets)
{
    volatile __shared__ unsigned int s_offset[BUCKET_BLOCK_MEMORY];

    int prefixBase = blockIdx.x * BUCKET_BLOCK_MEMORY;
    const int warpBase = (threadIdx.x >> BUCKET_WARP_LOG_SIZE) * DIVISIONS;
    const int numThreads = blockDim.x * gridDim.x;
    for (int i = threadIdx.x; i < BUCKET_BLOCK_MEMORY; i += blockDim.x)
        s_offset[i] = l_offsets[i & (DIVISIONS - 1)] + d_prefixoffsets[prefixBase + i];

    __syncthreads();

    for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < size; tid += numThreads) {

        float elem = input[tid];
        int id = indice[tid];

        output[s_offset[warpBase + (id & (DIVISIONS - 1))] + (id >> LOG_DIVISIONS)] = elem;
        int test = s_offset[warpBase + (id & (DIVISIONS - 1))] + (id >> LOG_DIVISIONS);
    }
}

#endif
