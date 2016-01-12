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
#ifndef GLOOP_HOST_CONTEXT_CU_H_
#define GLOOP_HOST_CONTEXT_CU_H_
#include <cuda.h>
#include <memory>
#include "device_context.cuh"
#include "host_loop.cuh"
#include "noncopyable.h"
namespace gloop {

class HostContext {
GLOOP_NONCOPYABLE(HostContext);
public:
    __host__ static std::unique_ptr<HostContext> create(HostLoop&, dim3 blocks);

    __host__ DeviceContext deviceContext() { return { m_context }; }

private:
    HostContext(dim3 blocks);
    bool initialize();

    void* m_context { nullptr };
    dim3 m_blocks { };
    bool m_resumed { false };
};

}  // namespace gloop
#endif  // GLOOP_HOST_CONTEXT_CU_H_