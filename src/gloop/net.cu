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

namespace tcp {

__device__ void connectImpl(DeviceLoop* loop, IPC* ipc, volatile request::NetTCPConnect& req, struct sockaddr_in* addr)
{
    GLOOP_ASSERT_SINGLE_THREAD();
    // FIXME: Fix this part.
    *const_cast<struct sockaddr_in*>(&req.address) = *reinterpret_cast<struct sockaddr_in*>(addr);
    ipc->emit(Code::NetTCPConnect);
}

__device__ void bindImpl(DeviceLoop* loop, IPC* ipc, volatile request::NetTCPBind& req, struct sockaddr_in* addr)
{
    GLOOP_ASSERT_SINGLE_THREAD();
    // FIXME: Fix this part.
    *const_cast<struct sockaddr_in*>(&req.address) = *reinterpret_cast<struct sockaddr_in*>(addr);
    ipc->emit(Code::NetTCPBind);
}

__device__ void acceptImpl(DeviceLoop* loop, IPC* ipc, volatile request::NetTCPAccept& req, net::Server* server)
{
    GLOOP_ASSERT_SINGLE_THREAD();
    req.server = server;
    ipc->emit(Code::NetTCPAccept);
}

__device__ void receiveImpl(DeviceLoop* loop, IPC* ipc, volatile request::NetTCPReceive& req, net::Socket* socket, size_t count, unsigned char* buffer)
{
    GLOOP_ASSERT_SINGLE_THREAD();
    req.socket = socket;
    req.count = count;
    req.buffer = buffer;
    ipc->emit(Code::NetTCPReceive);
}

__device__ void sendImpl(DeviceLoop* loop, IPC* ipc, volatile request::NetTCPSend& req, net::Socket* socket, size_t count, unsigned char* buffer)
{
    GLOOP_ASSERT_SINGLE_THREAD();
    req.socket = socket;
    req.count = count;
    req.buffer = buffer;
    ipc->emit(Code::NetTCPSend);
}

__device__ void closeImpl(DeviceLoop* loop, IPC* ipc, volatile request::NetClose& req, net::Socket* socket)
{
    GLOOP_ASSERT_SINGLE_THREAD();
    req.socket = socket;
    ipc->emit(Code::NetTCPClose);
}

}  // namespace gloop::net::tcp
} }  // namespace gloop::net
