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
#ifndef GLOOP_IPC_CU_H_
#define GLOOP_IPC_CU_H_
#include <cstdint>
#include <gipc/mapped.cuh>
#include "code.cuh"
#include "noncopyable.h"
#include "request.h"
#include "sync_read_write.h"
#include "utility.h"

namespace gloop {

class IPC : public gipc::Mapped {
GLOOP_NONCOPYABLE(IPC)
public:
    __host__ IPC() : m_request { 0 } { }
    __host__ __device__ void emit(Code code);
    __device__ __host__ Code peek();

    __device__ __host__ GLOOP_ALWAYS_INLINE volatile request::Request* request()
    {
        return &m_request;
    }

private:
#if 0
    __device__ void lock();
    __device__ void unlock();
#endif

    volatile request::Request m_request { 0 };
};

GLOOP_ALWAYS_INLINE __host__ __device__ void IPC::emit(Code code)
{
    syncWrite(&m_request.code, static_cast<int32_t>(code));
}

GLOOP_ALWAYS_INLINE __device__ __host__ Code IPC::peek()
{
    return static_cast<Code>(m_request.code);
}

}  // namespace gloop
#endif  // GLOOP_IPC_CU_H_
