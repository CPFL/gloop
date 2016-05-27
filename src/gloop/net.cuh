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
#include "device_loop_inlines.cuh"
#include "net_socket.h"
#include "request.h"
#include "utility/util.cu.h"

namespace gloop {
namespace net {
namespace tcp {

static_assert(sizeof(void*) == sizeof(uint64_t), "In both the host and the device, the size of the pointer should be 64bit.");

template<typename Lambda>
inline __device__ auto connect(DeviceLoop* loop, struct sockaddr_in* addr, Lambda callback) -> void
{
    BEGIN_SINGLE_THREAD
    {
        auto rpc = loop->enqueueRPC([callback](DeviceLoop* loop, volatile request::Request* req) {
            callback(loop, const_cast<Socket*>(req->u.netTCPConnectResult.socket));
        });
        volatile request::NetTCPConnect& req = rpc.request(loop)->u.netTCPConnect;
        *const_cast<struct sockaddr_in*>(&req.address) = *reinterpret_cast<struct sockaddr_in*>(addr);
        rpc.emit(loop, Code::NetTCPConnect);
    }
    END_SINGLE_THREAD
}

template<typename Lambda>
inline __device__ auto bind(DeviceLoop* loop, struct sockaddr_in* addr, Lambda callback) -> void
{
    BEGIN_SINGLE_THREAD
    {
        auto rpc = loop->enqueueRPC([callback](DeviceLoop* loop, volatile request::Request* req) {
            callback(loop, const_cast<Server*>(req->u.netTCPBindResult.server));
        });
        volatile request::NetTCPBind& req = rpc.request(loop)->u.netTCPBind;
        *const_cast<struct sockaddr_in*>(&req.address) = *reinterpret_cast<struct sockaddr_in*>(addr);
        rpc.emit(loop, Code::NetTCPBind);
    }
    END_SINGLE_THREAD
}

template<typename Lambda>
inline __device__ auto unbind(DeviceLoop* loop, Server* server, Lambda callback) -> void
{
    BEGIN_SINGLE_THREAD
    {
        auto rpc = loop->enqueueRPC([callback](DeviceLoop* loop, volatile request::Request* req) {
            callback(loop, req->u.netTCPUnbindResult.error);
        });
        volatile request::NetTCPUnbind& req = rpc.request(loop)->u.netTCPUnbind;
        req.server = server;
        rpc.emit(loop, Code::NetTCPUnbind);
    }
    END_SINGLE_THREAD
}

template<typename Lambda>
inline __device__ auto accept(DeviceLoop* loop, net::Server* server, Lambda callback) -> void
{
    BEGIN_SINGLE_THREAD
    {
        auto rpc = loop->enqueueRPC([callback](DeviceLoop* loop, volatile request::Request* req) {
            callback(loop, const_cast<Socket*>(req->u.netTCPAcceptResult.socket));
        });
        volatile request::NetTCPAccept& req = rpc.request(loop)->u.netTCPAccept;
        req.server = server;
        rpc.emit(loop, Code::NetTCPAccept);
    }
    END_SINGLE_THREAD
}

template<typename Lambda>
inline __device__ auto receiveOnePage(DeviceLoop* loop, net::Socket* socket, size_t count, int flags, Lambda callback) -> void
{
    GPU_ASSERT(count <= GLOOP_SHARED_PAGE_SIZE);
    loop->allocOnePage([=](DeviceLoop* loop, void* page) {
        BEGIN_SINGLE_THREAD
        {
            auto rpc = loop->enqueueRPC([=](DeviceLoop* loop, volatile request::Request* req) {
                __threadfence_system();
                callback(loop, req->u.netTCPReceiveResult.receiveCount, page);
            });
            volatile request::NetTCPReceive& req = rpc.request(loop)->u.netTCPReceive;
            req.socket = socket;
            req.count = count;
            req.buffer = static_cast<unsigned char*>(page);
            req.flags = flags;
            rpc.emit(loop, Code::NetTCPReceive);
        }
        END_SINGLE_THREAD
    });
}

template<typename Lambda>
inline __device__ auto performOnePageReceive(DeviceLoop* loop, net::Socket* socket, ssize_t requestedCount, int flags, size_t count, unsigned char* buffer, size_t requestedOffset, ssize_t receiveCount, void* page, Lambda callback) -> void
{
    ssize_t accumulatedCount = requestedOffset + receiveCount;

    GPU_ASSERT(receiveCount <= count);
    GPU_ASSERT(accumulatedCount <= count);
    if (receiveCount < 0) {
        callback(loop, -1);
        return;
    }

    bool nextCall = receiveCount != 0 && receiveCount == requestedCount && accumulatedCount != count;
    if (nextCall) {
        ssize_t requestedCount = min((count - accumulatedCount), GLOOP_SHARED_PAGE_SIZE);
        receiveOnePage(loop, socket, requestedCount, flags, [=](DeviceLoop* loop, ssize_t receiveCount, void* page) {
            performOnePageReceive(loop, socket, requestedCount, flags, count, buffer, accumulatedCount, receiveCount, page, callback);
        });
    }

    gpunet::copy_block_src_volatile(buffer + requestedOffset, reinterpret_cast<volatile uchar*>(page), receiveCount);
    BEGIN_SINGLE_THREAD
    {
        loop->freeOnePage(page);
    }
    END_SINGLE_THREAD

    if (!nextCall) {
        // Ensure buffer's modification is flushed.
        // __threadfence();
        callback(loop, accumulatedCount);
    }
}

template<typename Lambda>
inline __device__ auto receive(DeviceLoop* loop, net::Socket* socket, size_t count, unsigned char* buffer, int flags, Lambda callback) -> void
{
    ssize_t requestedCount = min(count, GLOOP_SHARED_PAGE_SIZE);
    receiveOnePage(loop, socket, requestedCount, flags, [=](DeviceLoop* loop, ssize_t receiveCount, void* page) {
        performOnePageReceive(loop, socket, requestedCount, flags, count, buffer, 0, receiveCount, page, callback);
    });
}

template<typename Lambda>
inline __device__ auto sendOnePage(DeviceLoop* loop, net::Socket* socket, size_t transferringSize, unsigned char* buffer, Lambda callback) -> void
{
    loop->allocOnePage([=](DeviceLoop* loop, void* page) {
        gpunet::copy_block_dst_volatile(reinterpret_cast<volatile uchar*>(page), buffer, transferringSize);
        BEGIN_SINGLE_THREAD
        {
            auto rpc = loop->enqueueRPC([=](DeviceLoop* loop, volatile request::Request* req) {
                BEGIN_SINGLE_THREAD
                {
                    loop->freeOnePage(page);
                }
                END_SINGLE_THREAD
                callback(loop, req->u.netTCPSendResult.sentCount);
            });
            volatile request::NetTCPSend& req = rpc.request(loop)->u.netTCPSend;
            req.socket = socket;
            req.count = transferringSize;
            req.buffer = static_cast<unsigned char*>(page);
            rpc.emit(loop, Code::NetTCPSend);
        }
        END_SINGLE_THREAD
    });
}

template<typename Lambda>
inline __device__ auto performOnePageSend(DeviceLoop* loop, net::Socket* socket, ssize_t requestedCount, size_t count, unsigned char* buffer, ssize_t requestedOffset, ssize_t sentCount, Lambda callback) -> void
{
    ssize_t accumulatedCount = requestedOffset + sentCount;

    GPU_ASSERT(sentCount <= count);
    GPU_ASSERT(accumulatedCount <= count);
    if (sentCount < 0) {
        callback(loop, -1);
        return;
    }

    bool nextCall = sentCount != 0 && sentCount == requestedCount && accumulatedCount != count;
    if (nextCall) {
        ssize_t requestedCount = min((count - accumulatedCount), GLOOP_SHARED_PAGE_SIZE);
        sendOnePage(loop, socket, requestedCount, buffer + accumulatedCount, [=](DeviceLoop* loop, ssize_t sentCount) {
            performOnePageSend(loop, socket, requestedCount, count, buffer, accumulatedCount, sentCount, callback);
        });
        return;
    }
    callback(loop, accumulatedCount);
}

template<typename Lambda>
inline __device__ auto send(DeviceLoop* loop, net::Socket* socket, size_t count, unsigned char* buffer, Lambda callback) -> void
{
    // __threadfence_system();
    ssize_t requestedCount = min(count, GLOOP_SHARED_PAGE_SIZE);
    sendOnePage(loop, socket, requestedCount, buffer, [=](DeviceLoop* loop, ssize_t sentCount) {
        performOnePageSend(loop, socket, requestedCount, count, buffer, 0, sentCount, callback);
    });
}

template<typename Lambda>
inline __device__ auto close(DeviceLoop* loop, Socket* socket, Lambda callback) -> void
{
    BEGIN_SINGLE_THREAD
    {
        auto rpc = loop->enqueueRPC([callback](DeviceLoop* loop, volatile request::Request* req) {
            callback(loop, req->u.netTCPCloseResult.error);
        });
        volatile request::NetTCPClose& req = rpc.request(loop)->u.netTCPClose;
        req.socket = socket;
        rpc.emit(loop, Code::NetTCPClose);
    }
    END_SINGLE_THREAD
}

}  // namespace gloop::net::tcp
} }  // namespace gloop::net
#endif  // GLOOP_NET_CU_H_
