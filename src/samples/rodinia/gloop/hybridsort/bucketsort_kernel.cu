/*
  Copyright (C) 2016 Yusuke Suzuki <yusuke.suzuki@sslab.ics.keio.ac.jp>

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
  ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
  THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <gloop/gloop.h>
#include "bucketsort.cuh"

typedef gloop::Global LoopType;

static __device__ void bucketsortKernel(gloop::DeviceLoop<LoopType>* loop, float* input, int* indice, float* output, int size, unsigned int* d_prefixoffsets, unsigned int* l_offsets)
{
    extern volatile __shared__ unsigned int s_offset[];

    int bid = loop->logicalBlockIdx().x;
    int prefixBase = bid * BUCKET_BLOCK_MEMORY;
    const int warpBase = (threadIdx.x >> BUCKET_WARP_LOG_SIZE) * DIVISIONS;
    const int numThreads = blockDim.x * loop->logicalGridDim().x;
    for (int i = threadIdx.x; i < BUCKET_BLOCK_MEMORY; i += blockDim.x)
        s_offset[i] = l_offsets[i & (DIVISIONS - 1)] + d_prefixoffsets[prefixBase + i];

    __syncthreads();

    for (int tid = bid * blockDim.x + threadIdx.x; tid < size; tid += numThreads) {

        float elem = input[tid];
        int id = indice[tid];

        output[s_offset[warpBase + (id & (DIVISIONS - 1))] + (id >> LOG_DIVISIONS)] = elem;
        int test = s_offset[warpBase + (id & (DIVISIONS - 1))] + (id >> LOG_DIVISIONS);
    }
}

void bucketsortGPU(gloop::HostLoop& hostLoop, gloop::HostContext& hostContext, dim3 blocks, dim3 threads, float* input, int* indice, float* output, int size, unsigned int* d_prefixoffsets, unsigned int* l_offsets)
{
    hostLoop.launchWithSharedMemory<LoopType>(hostContext, dim3(90), blocks, threads, sizeof(unsigned int) * BUCKET_BLOCK_MEMORY, [] __device__ (gloop::DeviceLoop<LoopType>* loop, float* input, int* indice, float* output, int size, unsigned int* d_prefixoffsets, unsigned int* l_offsets) {
        bucketsortKernel(loop, input, indice, output, size, d_prefixoffsets, l_offsets);
    }, input, indice, output, size, d_prefixoffsets, l_offsets);
}
