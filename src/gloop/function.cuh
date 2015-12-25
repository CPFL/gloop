/*
  Copyright (C) 2015 Yusuke Suzuki <yusuke.suzuki@sslab.ics.keio.ac.jp>

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
#ifndef GLOOP_FUNCTION_CU_H_
#define GLOOP_FUNCTION_CU_H_
#include <array>
#include <cstdint>
#include "gloop.cuh"
namespace gloop {

class DeviceLoop;

struct Callback {
    typedef void (*FunctionPtr)(void*, DeviceLoop* loop, int);
    __device__ Callback(FunctionPtr functionPtr)
        : m_functionPtr(functionPtr)
    {
    }

    __device__ void operator()(DeviceLoop* loop, int value)
    {
        m_functionPtr(this, loop, value);
    }

    FunctionPtr m_functionPtr;
};

template<typename RawLambda>
struct Lambda : public Callback {
    __device__ Lambda(const RawLambda& lambda)
        : Callback(Lambda::staticInvoke)
        , m_lambda(lambda)
    {
    }

    static __device__ void staticInvoke(void* ptr, DeviceLoop* loop, int value)
    {
        Lambda* lambda = static_cast<Lambda*>(ptr);
        lambda->m_lambda(loop, value);
    }

    RawLambda m_lambda;
};

}  // namespace gloop
#endif  // GLOOP_FUNCTION_CU_H_
