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
#include "matmul_server_config.h"

template <int BLOCK_SIZE>
__device__ void matrixMulCUDA(float *C, float *A, float *B, int wA, int wB, int bx, int by)
{
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

    // Declaration of the shared memory array As used to
    // store the sub-matrix of A
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

    // Declaration of the shared memory array Bs used to
    // store the sub-matrix of B
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
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

class MatMulServer {
public:
    __device__ MatMulServer(gloop::net::Server* server)
        : m_server(server)
    {
    }

    __device__ void accept(gloop::DeviceLoop<>* loop)
    {
        if (m_count++ != 10) {
            gloop::net::tcp::accept(loop, m_server, [=](gloop::DeviceLoop<>* loop, gloop::net::Socket* socket) {
                if (!socket) {
                    return;
                }
                this->handle(loop, socket);
            });
        }
    }

    __device__ void close(gloop::DeviceLoop<>* loop, gloop::net::Socket* socket)
    {
        gloop::net::tcp::close(loop, socket, [=](gloop::DeviceLoop<>* loop, int error) {
            this->accept(loop);
        });
    }

    template<typename Callback>
    __device__ void matmul(gloop::DeviceLoop<>* loop, int y, Callback callback)
    {
        float* lhs = m_message;
        float* rhs = m_message + MATRIX_SIZE;
        float* out = m_message + MATRIX_SIZE * 2;

        int xtimes = MATRIX_HW / blockDim.x;
        int ytimes = MATRIX_HW / blockDim.y;

        for (; y < ytimes; ++y) {
            for (int x = 0; x < xtimes; ++x) {
                matrixMulCUDA<SHARED_BLOCK_SIZE>(out, lhs, rhs, MATRIX_HW, MATRIX_HW, x, y);
            }

            if (gloop::loop::postTaskIfNecessary(loop, [=] (gloop::DeviceLoop<>* loop) {
                this->matmul(loop, y + 1, callback);
            })) {
                return;
            }
        }
        callback(loop);
    }

    __device__ void handle(gloop::DeviceLoop<>* loop, gloop::net::Socket* socket)
    {
        gloop::net::tcp::receive(loop, socket, MATRIX_SIZE * sizeof(float) * 2, (uint8_t*)m_message, MSG_WAITALL, [=](gloop::DeviceLoop<>* loop, ssize_t receiveCount) {
            if (receiveCount == 0) {
                this->close(loop, socket);
                return;
            }
            GPU_ASSERT(receiveCount == (MATRIX_SIZE * sizeof(float) * 2));
            matmul(loop, 0, [=] (gloop::DeviceLoop<>* loop) {
                gloop::net::tcp::send(loop, socket, MATRIX_SIZE * sizeof(float), (uint8_t*)(m_message + MATRIX_SIZE * 2), [=](gloop::DeviceLoop<>* loop, ssize_t sentCount) {
                    if (sentCount == 0) {
                        this->close(loop, socket);
                        return;
                    }
                    this->handle(loop, socket);
                });
            });
        });
    }

private:
    float m_message[MSG_SIZE];
    gloop::net::Server* m_server;
    int m_count { 0 };
};

__device__ gloop::net::Server* globalServer = nullptr;
__device__ volatile gpunet::INIT_LOCK initLock;
__device__ void gpuMain(gloop::DeviceLoop<>* loop, struct sockaddr_in* addr)
{
    __shared__ MatMulServer* matMulServer;
    __shared__ int toInit;
    BEGIN_SINGLE_THREAD
    {
        toInit = initLock.try_wait();
        if (toInit != 1)
            matMulServer = new MatMulServer(globalServer);
    }
    END_SINGLE_THREAD
    if (toInit == 1) {
        gloop::net::tcp::bind(loop, addr, [=](gloop::DeviceLoop<>* loop, gloop::net::Server* server) {
            assert(server);
            __shared__ MatMulServer* matMulServer;
            BEGIN_SINGLE_THREAD
            {
                globalServer = server;
                __threadfence();
                initLock.signal();
                matMulServer = new MatMulServer(globalServer);
            }
            END_SINGLE_THREAD
            matMulServer->accept(loop);
        });
        return;
    }
    matMulServer->accept(loop);
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
            CUDA_SAFE_CALL(cudaDeviceSetLimit(cudaLimitMallocHeapSize, (1 << 30)));
            gpunet_server_init(&addr, &dev_addr, argv[2]);
            printf("address:(%x),port:(%u)\n", ((struct sockaddr_in*)addr)->sin_addr.s_addr, ((struct sockaddr_in*)addr)->sin_port);
        } else {
            gpunet_usage_client(argc, argv);
            exit(1);
        }
    }

    gloop::Benchmark benchmark;
    benchmark.begin();
    {
        hostLoop->launch(*hostContext, blocks, THREADS_PER_TB, [=] GLOOP_DEVICE_LAMBDA (gloop::DeviceLoop<>* loop, struct sockaddr* address) {
            gpuMain(loop, (struct sockaddr_in*)address);
        }, dev_addr);
    }
    benchmark.end();
    printf("[%d] ", 0);
    benchmark.report();

    return 0;
}
