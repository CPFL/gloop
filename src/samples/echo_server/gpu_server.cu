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

#include <gloop/gloop.h>
#include <gloop/benchmark.h>
#include "microbench_util.h"

#define THREADS_PER_TB 256
#define BLOCKS 1
#define BUF_SIZE 65536
#define NR_MSG   60000
#define MSG_SIZE BUF_SIZE

__device__ unsigned char g_message[512][MSG_SIZE];

__device__ void accept(gloop::DeviceLoop* loop, gloop::net::Server* server);

__device__ void close(gloop::DeviceLoop* loop, gloop::net::Server* server, gloop::net::Socket* socket)
{
    gloop::net::tcp::close(loop, socket, [=](gloop::DeviceLoop* loop, int error) {
        accept(loop, server);
    });
}

__device__ void perform(gloop::DeviceLoop* loop, gloop::net::Server* server, gloop::net::Socket* socket)
{
    gloop::net::tcp::receive(loop, socket, BUF_SIZE, g_message[gloop::logicalBlockIdx.x], [=](gloop::DeviceLoop* loop, ssize_t receiveCount) {
        if (receiveCount == 0) {
            close(loop, server, socket);
            return;
        }
        gloop::net::tcp::send(loop, socket, receiveCount, g_message[gloop::logicalBlockIdx.x], [=](gloop::DeviceLoop* loop, ssize_t sentCount) {
            if (sentCount == 0) {
                close(loop, server, socket);
                return;
            }
            perform(loop, server, socket);
        });
    });
}

__device__ void accept(gloop::DeviceLoop* loop, gloop::net::Server* server)
{
    gloop::net::tcp::accept(loop, server, [=](gloop::DeviceLoop* loop, gloop::net::Socket* socket) {
        if (!socket) {
            return;
        }
        perform(loop, server, socket);
    });
}

__device__ gloop::net::Server* globalServer = nullptr;
__device__ volatile gpunet::INIT_LOCK initLock;
__device__ void gpuMain(gloop::DeviceLoop* loop, struct sockaddr_in* addr)
{
    BEGIN_SINGLE_THREAD
    {
        __shared__ int toInit;
        toInit = initLock.try_wait();
        if (toInit == 1) {
            gloop::net::tcp::bind(loop, addr, [=](gloop::DeviceLoop* loop, gloop::net::Server* server) {
                assert(server);
                BEGIN_SINGLE_THREAD
                {
                    globalServer = server;
                    __threadfence();
                    initLock.signal();
                }
                END_SINGLE_THREAD
                accept(loop, globalServer);
            });
            return;
        }
    }
    END_SINGLE_THREAD
    accept(loop, globalServer);
}

int main(int argc, char** argv)
{
    dim3 blocks(BLOCKS);
    std::unique_ptr<gloop::HostLoop> hostLoop = gloop::HostLoop::create(0);
    std::unique_ptr<gloop::HostContext> hostContext = gloop::HostContext::create(*hostLoop, blocks);

    struct sockaddr* addr;
    struct sockaddr* dev_addr;
    {
        if (argc > 2) {
            std::lock_guard<gloop::HostLoop::KernelLock> lock(hostLoop->kernelLock());
            CUDA_SAFE_CALL(cudaDeviceSetLimit(cudaLimitMallocHeapSize, (2 << 20) * 256));
            gpunet_client_init(&addr, &dev_addr, argv[1], argv[2]);
            printf("address:(%x),port:(%u)\n", ((struct sockaddr_in*)addr)->sin_addr.s_addr, ((struct sockaddr_in*)addr)->sin_port);
        } else {
            gpunet_usage_client(argc, argv);
            exit(1);
        }
    }

    gloop::Benchmark benchmark;
    benchmark.begin();
    {
        hostLoop->launch(*hostContext, THREADS_PER_TB, [=] GLOOP_DEVICE_LAMBDA (gloop::DeviceLoop* loop, struct sockaddr* address) {
            gpuMain(loop, (struct sockaddr_in*)address);
        }, dev_addr);
    }
    benchmark.end();
    printf("[%d] ", 0);
    benchmark.report();

    return 0;
}
