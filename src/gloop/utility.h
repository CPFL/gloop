/*
  Copyright (C) 2015-2016 Yusuke Suzuki <yusuke.suzuki@sslab.ics.keio.ac.jp>

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
#ifndef GLOOP_UTILITY_H_
#define GLOOP_UTILITY_H_
#include <assert.h>

#define GLOOP_CONCAT1(x, y) x##y
#define GLOOP_CONCAT(x, y) GLOOP_CONCAT1(x, y)

#define GLOOP_SINGLE_THREAD() \
    __syncthreads();\
    for (\
        bool GLOOP_CONCAT(context, __LINE__) { false };\
        threadIdx.x+threadIdx.y+threadIdx.z ==0 && (GLOOP_CONCAT(context, __LINE__) = !GLOOP_CONCAT(context, __LINE__));\
        __syncthreads()\
    )

// see http://www5d.biglobe.ne.jp/~noocyte/Programming/BigAlignmentBlock.html
#define GLOOP_ALIGNED_SIZE(size, alignment) ((size) + (alignment) - 1)
#define GLOOP_ALIGNED_ADDRESS(address, alignment) ((address + (alignment - 1)) & ~(alignment - 1))

// only 2^n and unsigned
#define GLOOP_ROUNDUP(x, y) (((x) + (y - 1)) & ~(y - 1))

// only 2^n and unsinged
#define GLOOP_ROUNDDOWN(x, y) ((x) & (-(y)))

#ifndef RELEASE
#define GLOOP_ASSERT(x) assert(x)
#else
#define GLOOP_ASSERT(x) do { } while (0)
#endif

#if defined(__NVCC__) && !defined(__clang__)
#define GLOOP_DEVICE_LAMBDA __device__
#else
#define GLOOP_DEVICE_LAMBDA
#endif

#define GLOOP_UNREACHABLE() GLOOP_ASSERT(0)

#define GLOOP_ASSERT_SINGLE_THREAD() GLOOP_ASSERT(threadIdx.x+threadIdx.y+threadIdx.z ==0)

#define GLOOP_CUDA_SAFE_CALL(x) if((x) != cudaSuccess) {\
        fprintf(stderr, "CUDA ERROR %s: %d %s\n", __FILE__, __LINE__, cudaGetErrorString(cudaGetLastError()));\
        exit(-1);\
    }

#if defined(__CUDACC__)
    #define GLOOP_ALWAYS_INLINE __forceinline__  /* inline __attribute__((__always_inline__)) */
#else
    #define GLOOP_ALWAYS_INLINE inline __attribute__((__always_inline__))
#endif

#define GLOOP_TID() (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)
#define GLOOP_TMAX() (blockDim.x * blockDim.y * blockDim.z)
#define GLOOP_BID() (blockIdx.x + blockIdx.y * gridDim.x)

// #define GLOOP_ERROR(str) do { __assert_fail(str,__FILE__,__LINE__,__func__); } while (0)
#define GLOOP_ERROR(str) GLOOP_ASSERT(0)

#endif  // GLOOP_UTILITY_H_
