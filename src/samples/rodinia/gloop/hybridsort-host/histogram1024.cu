/*
 * Copyright 1993-2007 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code
 * are hereby granted a nonexclusive, royalty-free license to use this code
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

#include "histogram1024.cuh"
#include <gloop/statistics.h>

///////////////////////////////////////////////////////////////////////////////
// This is nvidias histogram256 SDK example modded to do a 1024 point
// histogram
///////////////////////////////////////////////////////////////////////////////

//Total number of possible data values
#define BIN_COUNT 1024 // Changed from 256
#define HISTOGRAM_SIZE (BIN_COUNT * sizeof(unsigned int))
//Machine warp size
#ifndef __DEVICE_EMULATION__
//G80's warp size is 32 threads
#define WARP_LOG_SIZE 5
#else
//Emulation currently doesn't execute threads in coherent groups of 32 threads,
//which effectively means warp size of 1 thread for emulation modes
#define WARP_LOG_SIZE 0
#endif
//Warps in thread block
#define WARP_N 3
//Threads per block count
#ifdef HISTO_WG_SIZE_0
#define THREAD_N HISTO_WG_SIZE_0
#else
#define THREAD_N (WARP_N << WARP_LOG_SIZE)
#endif

//Per-block number of elements in histograms
#define BLOCK_MEMORY (WARP_N * BIN_COUNT)
#define IMUL(a, b) __mul24(a, b)

typedef gloop::Global LoopType;

GLOOP_VISIBILITY_HIDDEN
static __device__ void addData1024(volatile unsigned int* s_WarpHist, unsigned int data, unsigned int threadTag)
{
    unsigned int count;
    do {
        count = s_WarpHist[data] & 0x07FFFFFFU;
        count = threadTag | (count + 1);
        s_WarpHist[data] = count;
    } while (s_WarpHist[data] != count);
}

GLOOP_VISIBILITY_HIDDEN
static __device__ void performHistogram(gloop::DeviceLoop<LoopType>* loop, unsigned int* d_Result, float* d_Data, float minimum, float maximum, int dataN, int cursor)
{
    //Current global thread index
    const int globalTid = IMUL(loop->logicalBlockIdx().x, blockDim.x) + threadIdx.x;
    //Total number of threads in the compute grid
    const int numThreads = IMUL(blockDim.x, loop->logicalGridDim().x);
    //WARP_LOG_SIZE higher bits of counter values are tagged
    //by lower WARP_LOG_SIZE threadID bits
    // Will correctly issue warning when compiling for debug (x<<32-0)
    const unsigned int threadTag = threadIdx.x << (32 - WARP_LOG_SIZE);
    //Shared memory cache for each warp in current thread block
    //Declare as volatile to prevent incorrect compiler optimizations in addPixel()
    extern volatile __shared__ unsigned int s_Hist[];
    //Current warp shared memory frame
    const int warpBase = IMUL(threadIdx.x >> WARP_LOG_SIZE, BIN_COUNT);

    //Clear shared memory buffer for current thread block before processing
    for (int pos = threadIdx.x; pos < BLOCK_MEMORY; pos += blockDim.x)
        s_Hist[pos] = 0;

    __syncthreads();
    //Cycle through the entire data set, update subhistograms for each warp
    //Since threads in warps always execute the same instruction,
    //we are safe with the addPixel trick

    for (int pos = globalTid + cursor * numThreads;; pos += numThreads, ++cursor) {

        int result = pos < dataN;
        if (result) {
            unsigned int data4 = ((d_Data[pos] - minimum) / (maximum - minimum)) * BIN_COUNT;
            addData1024(s_Hist + warpBase, data4 & 0x3FFU, threadTag);
        }

        if (__syncthreads_and(!result)) {
            break;
        }

        if (gloop::loop::postTaskIfNecessary(loop,
            [=] (gloop::DeviceLoop<LoopType>* loop) {
                performHistogram(loop, d_Result, d_Data, minimum, maximum, dataN, cursor + 1);
            })) {
            //Merge per-warp histograms into per-block and write to global memory
            for (int pos = threadIdx.x; pos < BIN_COUNT; pos += blockDim.x) {
                unsigned int sum = 0;

                for (int base = 0; base < BLOCK_MEMORY; base += BIN_COUNT)
                    sum += s_Hist[base + pos] & 0x07FFFFFFU;
                atomicAdd(d_Result + pos, sum);
            }
            return;
        }
    }

    __syncthreads();
    //Merge per-warp histograms into per-block and write to global memory
    for (int pos = threadIdx.x; pos < BIN_COUNT; pos += blockDim.x) {
        unsigned int sum = 0;

        for (int base = 0; base < BLOCK_MEMORY; base += BIN_COUNT)
            sum += s_Hist[base + pos] & 0x07FFFFFFU;
        atomicAdd(d_Result + pos, sum);
    }
}

//Thread block (== subhistogram) count
#define BLOCK_N 64

////////////////////////////////////////////////////////////////////////////////
// Put all kernels together
////////////////////////////////////////////////////////////////////////////////
//histogram1024kernel() results buffer
static unsigned int* d_Result1024;

//Internal memory allocation
void initHistogram1024(void)
{
    checkCudaErrors(cudaMalloc((void**)&d_Result1024, HISTOGRAM_SIZE));
}

//Internal memory deallocation
void closeHistogram1024(void)
{
    checkCudaErrors(cudaFree(d_Result1024));
}

//histogram1024 CPU front-end
void histogram1024GPU(
    gloop::HostLoop& hostLoop,
    gloop::HostContext& hostContext,
    unsigned int* h_Result,
    float* d_Data,
    float minimum,
    float maximum,
    int dataN)
{
    {
        gloop::Statistics::Scope<gloop::Statistics::Type::Copy> scope;
        std::lock_guard<gloop::HostLoop::KernelLock> lock(hostLoop.kernelLock());
        checkCudaErrors(cudaMemset(d_Result1024, 0, HISTOGRAM_SIZE));
    }

    {
        gloop::Statistics::Scope<gloop::Statistics::Type::Kernel> scope;
        hostLoop.launchWithSharedMemory<LoopType>(hostContext, dim3(BLOCK_N), dim3(BLOCK_N), dim3(THREAD_N), sizeof(unsigned int) * BLOCK_MEMORY, [] __device__(
            gloop::DeviceLoop<LoopType>* loop,
            unsigned int* d_Result,
            float* d_Data,
            float minimum,
            float maximum,
            int dataN
            ) {
            performHistogram(loop, d_Result, d_Data, minimum, maximum, dataN, 0);
        }, d_Result1024, d_Data, minimum, maximum, dataN);
    }
    {
        gloop::Statistics::Scope<gloop::Statistics::Type::Copy> scope;
        std::lock_guard<gloop::HostLoop::KernelLock> lock(hostLoop.kernelLock());
        checkCudaErrors(cudaMemcpy(h_Result, d_Result1024, HISTOGRAM_SIZE, cudaMemcpyDeviceToHost));
    }
}
