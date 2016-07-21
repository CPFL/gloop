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
#ifndef GLOOP_HOST_LOOP_INLINES_CU_H_
#define GLOOP_HOST_LOOP_INLINES_CU_H_
#include "host_context.cuh"
#include "host_loop.cuh"
#include "data_log.h"
#include "entry.cuh"
namespace gloop {

template<typename DeviceLambda, class... Args>
inline void HostLoop::launch(HostContext& hostContext, dim3 logicalBlocks, dim3 threads, DeviceLambda&& callback, Args&&... args)
{
    launchWithSharedMemory<Shared>(hostContext, logicalBlocks, threads, 0, std::forward<DeviceLambda&&>(callback), std::forward<Args&&>(args)...);
}

template<typename DeviceLambda, class... Args>
inline void HostLoop::launch(HostContext& hostContext, dim3 physicalBlocks, dim3 logicalBlocks, dim3 threads, DeviceLambda&& callback, Args&&... args)
{
    launchWithSharedMemory<Shared>(hostContext, physicalBlocks, logicalBlocks, threads, 0, std::forward<DeviceLambda&&>(callback), std::forward<Args&&>(args)...);
}

template<typename Policy, typename DeviceLambda, class... Args>
inline void HostLoop::launch(HostContext& hostContext, dim3 logicalBlocks, dim3 threads, DeviceLambda&& callback, Args&&... args)
{
    launchWithSharedMemory<Policy>(hostContext, logicalBlocks, threads, 0, std::forward<DeviceLambda&&>(callback), std::forward<Args&&>(args)...);
}

template<typename Policy, typename DeviceLambda, class... Args>
inline void HostLoop::launch(HostContext& hostContext, dim3 physicalBlocks, dim3 logicalBlocks, dim3 threads, DeviceLambda&& callback, Args&&... args)
{
    launchWithSharedMemory<Policy>(hostContext, physicalBlocks, logicalBlocks, threads, 0, std::forward<DeviceLambda&&>(callback), std::forward<Args&&>(args)...);
}

template<typename Policy, typename DeviceLambda, class... Args>
inline void HostLoop::launchWithSharedMemory(HostContext& hostContext, dim3 logicalBlocks, dim3 threads, size_t sharedMemorySize, DeviceLambda&& callback, Args&&... args)
{
    launchWithSharedMemory<Policy>(hostContext, logicalBlocks, logicalBlocks, threads, sharedMemorySize, std::forward<DeviceLambda&&>(callback), std::forward<Args&&>(args)...);
}

template<typename Policy, typename DeviceLambda, class... Args>
inline void HostLoop::launchWithSharedMemory(HostContext& hostContext, dim3 preferredPhysicalBlocks, dim3 logicalBlocks, dim3 threads, size_t sharedMemorySize, DeviceLambda&& callback, Args&&... args)
{
    dim3 physicalBlocks = hostContext.maxPhysicalBlocks();
    uint64_t physicalBlocksNumber = physicalBlocks.x * physicalBlocks.y;
    uint64_t preferredPhysicalBlocksNumber = preferredPhysicalBlocks.x * preferredPhysicalBlocks.y;
    uint64_t logicalBlocksNumber = logicalBlocks.x * logicalBlocks.y;
    uint64_t resultBlocksNumber = preferredPhysicalBlocksNumber;

    resultBlocksNumber = std::min(physicalBlocksNumber, resultBlocksNumber);
    resultBlocksNumber = std::min(logicalBlocksNumber, resultBlocksNumber);

    return launchInternal<Policy>(hostContext, dim3(resultBlocksNumber), logicalBlocks, threads, sharedMemorySize, std::forward<DeviceLambda&&>(callback), std::forward<Args&&>(args)...);
}

template<typename Policy, typename DeviceLambda, class... Args>
inline void HostLoop::launchInternal(HostContext& hostContext, dim3 physicalBlocks, dim3 logicalBlocks, dim3 threads, size_t sharedMemorySize, DeviceLambda callback, Args... args)
{
//     std::shared_ptr<gloop::Benchmark> benchmark = std::make_shared<gloop::Benchmark>();
//     benchmark->begin();
    hostContext.prologue(logicalBlocks, physicalBlocks, sharedMemorySize);

#if 0
    {
        // Report occupancy.
        int minGridSize;
        int blockSize;
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, gloop::resume<DeviceLambda, Args...>, 0, 0);
        // GLOOP_DATA_LOG("grid:(%d),block:(%d)\n", minGridSize, blockSize);
    }
#endif

    {
        refKernel();
        m_kernelService.post([=, &hostContext] {
            {
                std::lock_guard<KernelLock> lock(m_kernelLock);
                // GLOOP_DATA_LOG("acquire for launch\n");
                prepareForLaunch(hostContext);
                gloop::resume<<<hostContext.physicalBlocks(), threads, hostContext.sharedMemorySize(), m_pgraph>>>(Policy::Policy, /* isInitialExecution */ 1, hostContext.deviceContext(), callback, args...);
                cudaError_t error = cudaGetLastError();
                GLOOP_CUDA_SAFE(error);
                GLOOP_CUDA_SAFE_CALL(cudaStreamSynchronize(m_pgraph));
            }

            if (hostContext.pending()) {
                resume<Policy>(hostContext, threads, callback, args...);
                return;
            }
            derefKernel();
        });
        drain();
    }

    hostContext.epilogue();
}

template<typename Policy, typename DeviceLambda, typename... Args>
void HostLoop::resume(HostContext& hostContext, dim3 threads, DeviceLambda callback, Args... args)
{
    // GLOOP_DEBUG("resume\n");
    m_kernelService.post([=, &hostContext] {
        bool acquireLockSoon = false;
        {
            {
                std::unique_lock<HostContext::Mutex> lock(hostContext.mutex());
                while (!hostContext.isReadyForResume(lock)) {
                    hostContext.condition().wait(lock);
                }
                m_kernelLock.lock();
            }
            // GLOOP_DATA_LOG("acquire for resume\n");
            prepareForLaunch(hostContext);

            {
                gloop::resume<<<hostContext.physicalBlocks(), threads, hostContext.sharedMemorySize(), m_pgraph>>>(Policy::Policy, /* isInitialExecution */ 0, hostContext.deviceContext(), callback, args...);
                cudaError_t error = cudaGetLastError();
                GLOOP_CUDA_SAFE(error);
                GLOOP_CUDA_SAFE_CALL(cudaStreamSynchronize(m_pgraph));
            }

            acquireLockSoon = hostContext.pending();

            // m_kernelLock.unlock(acquireLockSoon);
            // m_kernelLock.unlock();

            {
                // FIXME: Fix this.
                std::unique_lock<HostContext::Mutex> lock(hostContext.mutex());
                Command::ReleaseStatus releaseStatus = Command::ReleaseStatus::IO;
                if (hostContext.isReadyForResume(lock)) {
                    releaseStatus = Command::ReleaseStatus::Ready;
                }
                m_kernelLock.unlock(releaseStatus);
            }
        }
        if (acquireLockSoon) {
            resume<Policy>(hostContext, threads, callback, args...);
            return;
        }
        derefKernel();
    });
}

void HostLoop::lockLaunch()
{
    unsigned int priority { };
    std::size_t size { };
    Command command {
        .type = Command::Type::Lock,
        .payload = 0
    };
    m_requestQueue->send(&command, sizeof(Command), 0);
    m_responseQueue->receive(&command, sizeof(Command), size, priority);
}

void HostLoop::unlockLaunch(Command::ReleaseStatus releaseStatus)
{
    Command command {
        .type = Command::Type::Unlock,
        .payload = static_cast<uint64_t>(releaseStatus)
    };
    m_requestQueue->send(&command, sizeof(Command), 0);
}

}  // namespace gloop
#endif  // GLOOP_HOST_LOOP_INLINES_CU_H_
