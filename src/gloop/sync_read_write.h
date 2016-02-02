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
#ifndef GLOOP_SYNC_READ_WRITE_H_
#define GLOOP_SYNC_READ_WRITE_H_
#include "utility.h"
namespace gloop {

template<typename T, typename U>
#if defined(__CUDACC__)
__host__ __device__
#endif
GLOOP_ALWAYS_INLINE T readNoCache(volatile const U* ptr)
{
    return *reinterpret_cast<volatile const T*>(ptr);
}

template<typename T, typename U>
#if defined(__CUDACC__)
__host__ __device__
#endif
GLOOP_ALWAYS_INLINE void writeNoCache(volatile U* ptr, T value)
{
    *reinterpret_cast<volatile T*>(ptr) = value;
}

template<typename T>
#if defined(__CUDACC__)
__host__ __device__
#endif
GLOOP_ALWAYS_INLINE void syncWrite(volatile T* pointer, T value)
{
#if defined(__CUDA_ARCH__)
    __threadfence_system();
    writeNoCache<T>(pointer, value);
    __threadfence_system();
#else
    __sync_synchronize();
    writeNoCache<T>(pointer, value);
    __sync_synchronize();
#endif
}

}  // namespace gloop
#endif  // GLOOP_SYNC_READ_WRITE_H_
