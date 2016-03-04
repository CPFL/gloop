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
#ifndef GLOOP_NET_CU_H_
#define GLOOP_NET_CU_H_
#include <sys/types.h>
#include <sys/socket.h>
#include <type_traits>
#include <utility>
#include "device_loop.cuh"
#include "memcpy_io.cuh"
#include "net_socket.h"
#include "request.h"
#include "utility/util.cu.h"

namespace gloop {
namespace net {
namespace tcp {

__device__ void connectImpl(DeviceLoop* loop, IPC* ipc, volatile request::NetTCPConnect& req, struct sockaddr_in* addr);
__device__ void bindImpl(DeviceLoop* loop, IPC* ipc, volatile request::NetTCPBind& req, struct sockaddr_in* addr);
__device__ void unbindImpl(DeviceLoop* loop, IPC* ipc, volatile request::NetTCPUnbind& req, net::Server* server);
__device__ void acceptImpl(DeviceLoop* loop, IPC* ipc, volatile request::NetTCPAccept& req, net::Server* server);
__device__ void receiveImpl(DeviceLoop* loop, IPC* ipc, volatile request::NetTCPReceive& req, net::Socket* socket, size_t count, unsigned char* buffer);
__device__ void sendImpl(DeviceLoop* loop, IPC* ipc, volatile request::NetTCPSend& req, net::Socket* socket, size_t count, unsigned char* buffer);
__device__ void closeImpl(DeviceLoop* loop, IPC* ipc, volatile request::NetTCPClose& req, Socket* socket);

template<typename Lambda>
inline __device__ auto connect(DeviceLoop* loop, struct sockaddr_in* addr, Lambda callback) -> void
{
    BEGIN_SINGLE_THREAD
    {
        auto* ipc = loop->enqueueIPC([callback](DeviceLoop* loop, volatile request::Request* req) {
            callback(loop, const_cast<Socket*>(req->u.netTCPConnectResult.socket));
        });
        tcp::connectImpl(loop, ipc, ipc->request()->u.netTCPConnect, addr);
    }
    END_SINGLE_THREAD
}

template<typename Lambda>
inline __device__ auto bind(DeviceLoop* loop, struct sockaddr_in* addr, Lambda callback) -> void
{
    BEGIN_SINGLE_THREAD
    {
        auto* ipc = loop->enqueueIPC([callback](DeviceLoop* loop, volatile request::Request* req) {
            callback(loop, const_cast<Server*>(req->u.netTCPBindResult.server));
        });
        tcp::bindImpl(loop, ipc, ipc->request()->u.netTCPBind, addr);
    }
    END_SINGLE_THREAD
}

template<typename Lambda>
inline __device__ auto unbind(DeviceLoop* loop, Server* server, Lambda callback) -> void
{
    BEGIN_SINGLE_THREAD
    {
        auto* ipc = loop->enqueueIPC([callback](DeviceLoop* loop, volatile request::Request* req) {
            callback(loop, req->u.netTCPUnbindResult.error);
        });
        tcp::unbindImpl(loop, ipc, ipc->request()->u.netTCPUnbind, server);
    }
    END_SINGLE_THREAD
}

template<typename Lambda>
inline __device__ auto accept(DeviceLoop* loop, net::Server* server, Lambda callback) -> void
{
    BEGIN_SINGLE_THREAD
    {
        auto* ipc = loop->enqueueIPC([callback](DeviceLoop* loop, volatile request::Request* req) {
            callback(loop, const_cast<Socket*>(req->u.netTCPAcceptResult.socket));
        });
        tcp::acceptImpl(loop, ipc, ipc->request()->u.netTCPAccept, server);
    }
    END_SINGLE_THREAD
}

template<typename Lambda>
inline __device__ auto receive(DeviceLoop* loop, net::Socket* socket, size_t count, unsigned char* buffer, Lambda callback) -> void
{
    GPU_ASSERT(count <= GLOOP_SHARED_PAGE_SIZE);
    BEGIN_SINGLE_THREAD
    {
        loop->allocOnePage([=](DeviceLoop* loop, volatile request::Request* req) {
            BEGIN_SINGLE_THREAD
            {
                void* page = req->u.allocOnePageResult.page;
                auto* ipc = loop->enqueueIPC([=](DeviceLoop* loop, volatile request::Request* req) {
                    ssize_t receiveCount = req->u.netTCPReceiveResult.receiveCount;
                    __threadfence_system();
                    GPU_ASSERT(receiveCount <= GLOOP_SHARED_PAGE_SIZE);
                    gpunet::copy_block_src_volatile(buffer, reinterpret_cast<volatile uchar*>(page), receiveCount);
                    BEGIN_SINGLE_THREAD
                    {
                        loop->freeOnePage(page);
                    }
                    END_SINGLE_THREAD
                    callback(loop, receiveCount);
                });
                tcp::receiveImpl(loop, ipc, ipc->request()->u.netTCPReceive, socket, count, static_cast<unsigned char*>(page));
            }
            END_SINGLE_THREAD
        });
    }
    END_SINGLE_THREAD
}

template<typename Lambda>
inline __device__ auto send(DeviceLoop* loop, net::Socket* socket, size_t count, unsigned char* buffer, Lambda callback) -> void
{
    GPU_ASSERT(count <= GLOOP_SHARED_PAGE_SIZE);
    BEGIN_SINGLE_THREAD
    {
        loop->allocOnePage([=](DeviceLoop* loop, volatile request::Request* req) {
            unsigned char* page = static_cast<unsigned char*>(req->u.allocOnePageResult.page);
            gpunet::copy_block_dst_volatile(reinterpret_cast<volatile uchar*>(page), buffer, count);
            BEGIN_SINGLE_THREAD
            {
                auto* ipc = loop->enqueueIPC([=](DeviceLoop* loop, volatile request::Request* req) {
                    BEGIN_SINGLE_THREAD
                    {
                        loop->freeOnePage(page);
                    }
                    END_SINGLE_THREAD
                    callback(loop, req->u.netTCPSendResult.sentCount);
                });
                tcp::sendImpl(loop, ipc, ipc->request()->u.netTCPSend, socket, count, page);
            }
            END_SINGLE_THREAD
        });
    }
    END_SINGLE_THREAD
}

template<typename Lambda>
inline __device__ auto close(DeviceLoop* loop, Socket* socket, Lambda callback) -> void
{
    BEGIN_SINGLE_THREAD
    {
        auto* ipc = loop->enqueueIPC([callback](DeviceLoop* loop, volatile request::Request* req) {
            callback(loop, req->u.netTCPCloseResult.error);
        });
        tcp::closeImpl(loop, ipc, ipc->request()->u.netTCPClose, socket);
    }
    END_SINGLE_THREAD
}

}  // namespace gloop::net::tcp
} }  // namespace gloop::net
#endif  // GLOOP_NET_CU_H_
