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
#include <gloop/statistics.h>
#include "bucketsort.cuh"

static __global__ void bucketprefixoffsetKernel(unsigned int* d_prefixoffsets, unsigned int* d_offsets, int blocks)
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

void bucketprefixoffsetGPU(gloop::HostLoop& hostLoop, gloop::HostContext& hostContext, dim3 blocks, dim3 threads, unsigned int* d_prefixoffsets, unsigned int* d_offsets, int aBlocks)
{
    gloop::Statistics::Scope<gloop::Statistics::Type::Kernel> scope;
    std::lock_guard<gloop::HostLoop::KernelLock> lock(hostLoop.kernelLock());
    bucketprefixoffsetKernel<<<blocks, threads, 0, hostLoop.pgraph()>>>(d_prefixoffsets, d_offsets, aBlocks);
    GLOOP_CUDA_SAFE_CALL(cudaStreamSynchronize(hostLoop.pgraph()));
}
