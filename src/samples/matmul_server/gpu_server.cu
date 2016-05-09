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
#define MATRIX_HW 1024
#define MATRIX_SIZE (MATRIX_HW * MATRIX_HW)
#define MSG_SIZE (MATRIX_SIZE * 3)

template <int BLOCK_SIZE>
__device__ void matrixMulCUDA(float *C, float *A, float *B, int wA, int wB)
{
    // Block index
    int bx = gloop::logicalBlockIdx.x;
    int by = gloop::logicalBlockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
        //#pragma unroll

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Csub += As[ty][k] * Bs[k][tx];
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}

__device__ float g_message[BLOCKS][MSG_SIZE];

__device__ void accept(gloop::DeviceLoop* loop, gloop::net::Server* server);

__device__ void close(gloop::DeviceLoop* loop, gloop::net::Server* server, gloop::net::Socket* socket)
{
    gloop::net::tcp::close(loop, socket, [=](gloop::DeviceLoop* loop, int error) {
        accept(loop, server);
    });
}

__device__ void perform(gloop::DeviceLoop* loop, gloop::net::Server* server, gloop::net::Socket* socket)
{
    gloop::net::tcp::receive(loop, socket, MATRIX_SIZE * 2, (uint8_t*)(g_message[gloop::logicalBlockIdx.x]), 0, [=](gloop::DeviceLoop* loop, ssize_t receiveCount) {
        if (receiveCount == 0) {
            close(loop, server, socket);
            return;
        }
        float* lhs = g_message[gloop::logicalBlockIdx.x];
        float* rhs = g_message[gloop::logicalBlockIdx.x] + MATRIX_SIZE;
        float* out = g_message[gloop::logicalBlockIdx.x] + MATRIX_SIZE * 2;
        matrixMulCUDA<1>(out, lhs, rhs, MATRIX_HW, MATRIX_HW);
        gloop::net::tcp::send(loop, socket, receiveCount, (uint8_t*)out, [=](gloop::DeviceLoop* loop, ssize_t sentCount) {
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
