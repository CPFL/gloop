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
#include <cassert>
#include <mutex>
#include "data_log.h"
#include "device_context.cuh"
#include "rpc.cuh"
#include "host_context.cuh"
#include "host_loop_inlines.cuh"
#include "make_unique.h"
#include "sync_read_write.h"

namespace gloop {

std::unique_ptr<HostContext> HostContext::create(HostLoop& hostLoop, dim3 physicalBlocks, uint32_t pageCount)
{
    std::unique_ptr<HostContext> hostContext(new HostContext(hostLoop, physicalBlocks, pageCount));
    if (!hostContext->initialize(hostLoop)) {
        return nullptr;
    }
    return hostContext;
}

HostContext::HostContext(HostLoop& hostLoop, dim3 physicalBlocks, uint32_t pageCount)
    : m_hostLoop(hostLoop)
    , m_maxPhysicalBlocks(physicalBlocks)
    , m_physicalBlocks(sumOfBlocks(physicalBlocks))
    , m_pageCount(pageCount)
{
    // fprintf(stderr, "physical blocks %u\n", sumOfBlocks(physicalBlocks));
}

HostContext::~HostContext()
{
    // GLOOP_DATA_LOG("let's cleanup context\n");
    if (m_poller) {
        m_poller->interrupt();
        m_poller->join();
        m_poller.reset();
    }
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

        size_t blocksSize = m_maxPhysicalBlocks.x * m_maxPhysicalBlocks.y;
        size_t allSlotsSize = blocksSize * GLOOP_SHARED_SLOT_SIZE;

        m_codesMemory = MappedMemory::create(sizeof(int32_t) * allSlotsSize);
        m_codes = (int32_t*)m_codesMemory->mappedPointer();

        m_payloadsMemory = MappedMemory::create(sizeof(request::Payload) * allSlotsSize);
        m_payloads = (request::Payload*)m_payloadsMemory->mappedPointer();

        m_hostContextMemory = MappedMemory::create(sizeof(PerBlockHostContext) * blocksSize);
        m_hostContext = (PerBlockHostContext*)m_hostContextMemory->mappedPointer();

        m_kernel = MappedMemory::create(sizeof(KernelContext));

        m_context.killClock = hostLoop.killClock();

        GLOOP_CUDA_SAFE_CALL(cudaHostGetDevicePointer(&m_context.codes, m_codesMemory->mappedPointer(), 0));
        GLOOP_CUDA_SAFE_CALL(cudaHostGetDevicePointer(&m_context.payloads, m_payloadsMemory->mappedPointer(), 0));
        GLOOP_CUDA_SAFE_CALL(cudaHostGetDevicePointer(&m_context.hostContext, m_hostContextMemory->mappedPointer(), 0));
        GLOOP_CUDA_SAFE_CALL(cudaHostGetDevicePointer(&m_context.kernel, m_kernel->mappedPointer(), 0));
        // fprintf(stderr, "SIZE %u / %u\n", sizeof(PerBlockContext) * blocksSize, sizeof(PerBlockContext));
        GLOOP_CUDA_SAFE_CALL(cudaMalloc(&m_context.context, sizeof(PerBlockContext) * blocksSize));
        GLOOP_CUDA_SAFE_CALL(cudaMalloc(&m_context.deviceLoopStorage, sizeof(UninitializedDeviceLoopStorage) * blocksSize));
        if (m_pageCount) {
            GLOOP_CUDA_SAFE_CALL(cudaMalloc(&m_context.pages, sizeof(OnePage) * m_pageCount * blocksSize));
        }

        assert(!m_poller);
        m_poller = make_unique<boost::thread>([this]() {
            m_hostLoop.initializeInThread();
            pollerMain();
        });
    }
    return true;
}

uint32_t HostContext::pending() const
{
    return readNoCache<uint32_t>(&((KernelContext*)m_kernel->mappedPointer())->pending);
}

bool HostContext::isReadyForResume(const std::unique_lock<Mutex>&)
{
    if (!m_exitRequired.empty()) {
        return true;
    }

#if defined(GLOOP_ENABLE_IO_BOOSTING)
// TODO: Without elastic kernels, this state becomes super large.
// To avoid this situation, we need to stop the device loop from the host loop.
// At that time, we should lock the context to prevent event completion.
#if !defined(GLOOP_ENABLE_ELASTIC_KERNELS)
#error "Elastic kernels are needed to enable I/O boosting."
#endif
    uint64_t blocks = m_physicalBlocks;
    for (uint64_t i = 0; i < blocks; ++i) {
        PerBlockHostContext hostContext = m_hostContext[i];

        // FIXME: This is not correct in the last sequence (some TBs are already finished).
        // But, this may indicate that the next logical TB will start.
        if (hostContext.freeSlots == DeviceLoopControl::allFilledFreeSlots()) {
            return true;
        }

        uint32_t allocatedSlots = hostContext.freeSlots ^ DeviceLoopControl::allFilledFreeSlots();
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

                RPC rpc { i * GLOOP_SHARED_SLOT_SIZE + j };
                Code code = rpc.peek(*this);
                if (code == Code::Complete) {
                    return true;
                }
            }
        }
    }
    return false;
#else
    return true;
#endif
}

void HostContext::prepareForLaunch()
{
    writeNoCache<uint64_t>(&((KernelContext*)m_kernel->mappedPointer())->globalClock, 0);
    writeNoCache<uint32_t>(&((KernelContext*)m_kernel->mappedPointer())->pending, 0);
    // Clean up ExitRequired flags.
    {
        std::unique_lock<Mutex> guard(m_mutex);
        for (RPC rpc : m_exitRequired) {
            rpc.emit(*this, Code::Complete);
        }
        m_exitRequired.clear();
        for (std::shared_ptr<MmapResult> result : m_unmapRequests) {
            if (!result->refCount) {
                // GLOOP_DATA_LOG("Unmapping. fd:(%d),offset:(%llx),size:(%llx)\n", std::get<0>(result->request), std::get<1>(result->request), std::get<2>(result->request));
                GLOOP_CUDA_SAFE_CALL(cudaHostUnregister(result->device));
                ::munmap((char*)result->host, result->size);
                table().dropMmapResult(result);
            }
        }
        m_unmapRequests.clear();
    }
    __sync_synchronize();
}

void HostContext::pollerMain()
{
    uint32_t count = 0;
    while (true) {
        bool found = tryPeekRequest([&](RPC rpc) {
            request::Request req { };
            Code code = rpc.peek(*this);
            memcpy(&req, (request::Request*)rpc.request(*this), sizeof(request::Request));
            {
                std::lock_guard<HostContext::Mutex> lock(mutex());
                rpc.emit(*this, Code::Handling);
                condition().notify_one();
            }
            m_hostLoop.handleIO(*this, rpc, code, req);
        });
        if (found) {
            count = 0;
            continue;
        }
        // if ((++count % 100000) == 0) {
            boost::this_thread::interruption_point();
        // }
    }
}

void HostContext::prologue(dim3 logicalBlocks, dim3 physicalBlocks)
{
    assert((physicalBlocks.x * physicalBlocks.y) <= (m_maxPhysicalBlocks.x * m_maxPhysicalBlocks.y));
    m_physicalBlocks = sumOfBlocks(physicalBlocks);
    m_logicalBlocks = logicalBlocks;
    m_context.logicalBlocks = m_logicalBlocks;
}

void HostContext::epilogue()
{
    m_physicalBlocks = 1;
    m_logicalBlocks = dim3();
}

}  // namespace gloop
