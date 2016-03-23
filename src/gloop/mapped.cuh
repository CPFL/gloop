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
#ifndef GLOOP_MAPPED_CU_H_
#define GLOOP_MAPPED_CU_H_
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdio>
#include "utility.h"
namespace gloop {

class Mapped {
public:
    inline void* operator new(std::size_t size)
    {
        void* result = nullptr;
        GLOOP_CUDA_SAFE_CALL(cudaHostAlloc(&result, size, cudaHostAllocMapped));
        return result;
    }

    inline void* operator new[](std::size_t size)
    {
        void* result = nullptr;
        GLOOP_CUDA_SAFE_CALL(cudaHostAlloc(&result, size, cudaHostAllocMapped));
        return result;
    }

    inline void operator delete(void* ptr)
    {
        GLOOP_CUDA_SAFE_CALL(cudaFreeHost(ptr));
    }

    inline void operator delete[](void* ptr)
    {
        GLOOP_CUDA_SAFE_CALL(cudaFreeHost(ptr));
    }
};

}  // namespace gloop
#endif  // GLOOP_MAPPED_CU_H_
