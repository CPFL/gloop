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

#include <cstdint>
#include "code.cuh"
#include "device_loop.cuh"
#include "net.cuh"
#include "memcpy_io.cuh"
#include "request.h"

namespace gloop {
namespace net {

static_assert(sizeof(void*) == sizeof(uint64_t), "In both the host and the device, the size of the pointer should be 64bit.");

__device__ void socketImpl(DeviceLoop* loop, IPC* ipc, volatile request::NetSocket& req, int domain, int type, int protocol)
{
    GLOOP_ASSERT_SINGLE_THREAD();
    req.domain = domain;
    req.type = type;
    req.protocol = protocol;
    ipc->emit(Code::NetSocket);
}

__device__ void closeImpl(DeviceLoop* loop, IPC* ipc, volatile request::NetClose& req, Socket* socket)
{
    GLOOP_ASSERT_SINGLE_THREAD();
    req.socket = socket;
    ipc->emit(Code::NetClose);
}

} }  // namespace gloop::net
