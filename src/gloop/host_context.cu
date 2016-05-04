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
#include <mutex>
#include "data_log.h"
#include "device_context.cuh"
#include "ipc.cuh"
#include "host_context.cuh"
#include "host_loop.cuh"
#include "make_unique.h"
#include "sync_read_write.h"

namespace gloop {

static bool isZeroBlocks(dim3 blocks)
{
    return (blocks.x * blocks.y) == 0;
}

std::unique_ptr<HostContext> HostContext::create(HostLoop& hostLoop, dim3 logicalBlocks, dim3 physicalBlocks, uint32_t pageCount)
{
    std::unique_ptr<HostContext> hostContext(new HostContext(hostLoop, logicalBlocks, physicalBlocks, pageCount));
    if (!hostContext->initialize(hostLoop)) {
        return nullptr;
    }
    return hostContext;
}

HostContext::HostContext(HostLoop& hostLoop, dim3 logicalBlocks, dim3 physicalBlocks, uint32_t pageCount)
    : m_hostLoop(hostLoop)
    , m_logicalBlocks(logicalBlocks)
    , m_physicalBlocks(isZeroBlocks(physicalBlocks) ? logicalBlocks : physicalBlocks)
    , m_pageCount(pageCount)
{
}

HostContext::~HostContext()
{
    // GLOOP_DATA_LOG("let's cleanup context\n");
    {
        std::lock_guard<gloop::HostLoop::KernelLock> lock(m_hostLoop.kernelLock());
        // GLOOP_DATA_LOG("let's cleanup context lock acquire\n");
        if (m_context.context) {
            cudaFree(m_context.context);
        }
        m_codesMemory.reset();
        m_payloadsMemory.reset();
        m_kernel.reset();
    }
    // GLOOP_DATA_LOG("let's cleanup context done\n");
}

bool HostContext::initialize(HostLoop& hostLoop)
{
    {
        std::lock_guard<gloop::HostLoop::KernelLock> lock(hostLoop.kernelLock());

        size_t blocksSize = m_physicalBlocks.x * m_physicalBlocks.y;
        size_t allSlotsSize = blocksSize * GLOOP_SHARED_SLOT_SIZE;

        m_codesMemory = MappedMemory::create(sizeof(int32_t) * allSlotsSize);
        m_codes = (int32_t*)m_codesMemory->mappedPointer();

        m_payloadsMemory = MappedMemory::create(sizeof(request::Payload) * allSlotsSize);
        m_payloads = (request::Payload*)m_payloadsMemory->mappedPointer();

        m_hostContextMemory = MappedMemory::create(sizeof(DeviceContext::PerBlockHostContext) * blocksSize);
        m_hostContext = (DeviceContext::PerBlockHostContext*)m_hostContextMemory->mappedPointer();

        m_kernel = MappedMemory::create(sizeof(DeviceContext::KernelContext));

        m_context.killClock = hostLoop.killClock();
        m_context.logicalBlocks = m_logicalBlocks;

        GLOOP_CUDA_SAFE_CALL(cudaHostGetDevicePointer(&m_context.codes, m_codesMemory->mappedPointer(), 0));
        GLOOP_CUDA_SAFE_CALL(cudaHostGetDevicePointer(&m_context.payloads, m_payloadsMemory->mappedPointer(), 0));
        GLOOP_CUDA_SAFE_CALL(cudaHostGetDevicePointer(&m_context.hostContext, m_hostContextMemory->mappedPointer(), 0));
        GLOOP_CUDA_SAFE_CALL(cudaHostGetDevicePointer(&m_context.kernel, m_kernel->mappedPointer(), 0));

        GLOOP_CUDA_SAFE_CALL(cudaMalloc(&m_context.context, sizeof(DeviceContext::PerBlockContext) * blocksSize));
        if (m_pageCount) {
            GLOOP_CUDA_SAFE_CALL(cudaMalloc(&m_context.pages, sizeof(DeviceContext::OnePage) * m_pageCount * blocksSize));
        }
    }
    return true;
}

uint32_t HostContext::pending() const
{
    return readNoCache<uint32_t>(&((DeviceContext::KernelContext*)m_kernel->mappedPointer())->pending);
}

bool HostContext::isReadyForResume(const std::unique_lock<Mutex>&)
{
    if (!m_exitRequired.empty()) {
        return true;
    }

    int blocks = m_physicalBlocks.x * m_physicalBlocks.y;
    for (int i = 0; i < blocks; ++i) {
        DeviceContext::PerBlockHostContext hostContext = m_hostContext[i];
        if (hostContext.freeSlots == DeviceContext::DeviceLoopControl::allFilledFreeSlots()) {
            continue;
        }

        uint32_t allocatedSlots = hostContext.freeSlots ^ DeviceContext::DeviceLoopControl::allFilledFreeSlots();
        for (uint32_t j = 0; j < GLOOP_SHARED_SLOT_SIZE && allocatedSlots; ++j) {
            uint32_t bit = 1U << j;
            if (allocatedSlots & bit) {
                allocatedSlots &= ~bit;
                if (hostContext.sleepSlots & bit) {
                    if (hostContext.wakeupSlots & bit) {
                        return true;
                    }
                    continue;
                }

                IPC ipc { i * GLOOP_SHARED_SLOT_SIZE + j };
                Code code = ipc.peek(this);
                if (code == Code::Complete) {
                    return true;
                }
            }
        }
    }
    return false;
}

void HostContext::prepareForLaunch()
{
    writeNoCache<uint64_t>(&((DeviceContext::KernelContext*)m_kernel->mappedPointer())->globalClock, 0);
    writeNoCache<uint32_t>(&((DeviceContext::KernelContext*)m_kernel->mappedPointer())->pending, 0);
    // Clean up ExitRequired flags.
    {
        std::unique_lock<Mutex> guard(m_mutex);
        for (IPC ipc : m_exitRequired) {
            ipc.emit(this, Code::Complete);
        }
        m_exitRequired.clear();
        for (std::shared_ptr<MmapResult> result : m_unmapRequests) {
            if (!result->refCount) {
                GLOOP_CUDA_SAFE_CALL(cudaHostUnregister(result->device));
                ::munmap(((char*)result->host) + std::get<1>(result->request), result->size);
                table().dropMmapResult(result);
            }
        }
        m_unmapRequests.clear();
    }
    __sync_synchronize();
}

}  // namespace gloop
