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
#include "mergesort.cuh"
#include "mergesort_inlines.cuh"

static __global__ void mergeSortFirstKernel(float4* result, int listsize)
{
    // Block index
    int bx = blockIdx.x;
    // Thread index
    //int tx = threadIdx.x;
    if (bx * blockDim.x + threadIdx.x < listsize / 4) {
        float4 r = tex1Dfetch(tex, (int)(bx * blockDim.x + threadIdx.x));
        result[bx * blockDim.x + threadIdx.x] = sortElem(r);
    }
}

void mergeSortFirst(Context* ctx, dim3 grid, dim3 threads, float4* result, int listsize)
{
    gloop::Statistics::Scope<gloop::Statistics::Type::Kernel> scope;
    mergeSortFirstKernel<<<grid, threads>>>(result, listsize);
    cudaThreadSynchronize();
}
