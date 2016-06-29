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

#include "bucketsort.cuh"
#include "helper_cuda.h"
#include <GL/glew.h>
#include <GL/glut.h>
#include <cuda_gl_interop.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <gloop/gloop.h>

texture<float, 1, cudaReadModeElementType> texPivot;

static __device__ int addOffset(volatile unsigned int* s_offset, unsigned int data, unsigned int threadTag)
{
    unsigned int count;

    do {
        count = s_offset[data] & 0x07FFFFFFU;
        count = threadTag | (count + 1);
        s_offset[data] = count;
    } while (s_offset[data] != count);

    return (count & 0x07FFFFFFU) - 1;
}

static __device__ void
bucketcountKernel(gloop::DeviceLoop<gloop::Global>* loop, float* input, int* indice, unsigned int* d_prefixoffsets, int size)
{
    volatile __shared__ unsigned int s_offset[BUCKET_BLOCK_MEMORY];
//     __shared__ volatile unsigned int* s_offset;
//     BEGIN_SINGLE_THREAD
//         s_offset = new unsigned int[BUCKET_BLOCK_MEMORY];
//     END_SINGLE_THREAD

    const unsigned int threadTag = threadIdx.x << (32 - BUCKET_WARP_LOG_SIZE);
    const int warpBase = (threadIdx.x >> BUCKET_WARP_LOG_SIZE) * DIVISIONS;
    const int numThreads = blockDim.x * loop->logicalGridDim().x;
    for (int i = threadIdx.x; i < BUCKET_BLOCK_MEMORY; i += blockDim.x)
        s_offset[i] = 0;

    __syncthreads();

    for (int tid = loop->logicalBlockIdx().x * blockDim.x + threadIdx.x; tid < size; tid += numThreads) {
        float elem = input[tid];

        int idx = DIVISIONS / 2 - 1;
        int jump = DIVISIONS / 4;
        float piv = tex1Dfetch(texPivot, idx); //s_pivotpoints[idx];

        while (jump >= 1) {
            idx = (elem < piv) ? (idx - jump) : (idx + jump);
            piv = tex1Dfetch(texPivot, idx); //s_pivotpoints[idx];
            jump /= 2;
        }
        idx = (elem < piv) ? idx : (idx + 1);

        indice[tid] = (addOffset(s_offset + warpBase, idx, threadTag) << LOG_DIVISIONS) + idx; //atomicInc(&offsets[idx], size + 1);
    }

    __syncthreads();

    int prefixBase = loop->logicalBlockIdx().x * BUCKET_BLOCK_MEMORY;

    for (int i = threadIdx.x; i < BUCKET_BLOCK_MEMORY; i += blockDim.x)
        d_prefixoffsets[prefixBase + i] = s_offset[i] & 0x07FFFFFFU;


//     BEGIN_SINGLE_THREAD
//         delete [] s_offset;
//     END_SINGLE_THREAD
}

void bucketcount(gloop::HostLoop& hostLoop, gloop::HostContext& hostContext, dim3 grid, dim3 threads, float* input, int* indice, unsigned int* d_prefixoffsets, int size)
{
    hostLoop.launch<gloop::Global>(hostContext, dim3(360), grid, threads, [] __device__ (gloop::DeviceLoop<gloop::Global>* loop, float* input, int* indice, unsigned int* d_prefixoffsets, int size) {
        bucketcountKernel(loop, input, indice, d_prefixoffsets, size);
    }, input, indice, d_prefixoffsets, size);
    // bucketcount<<<grid, threads>>>(d_input, d_indice, d_prefixoffsets, listsize);
}
