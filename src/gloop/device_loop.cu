/*
  Copyright (C) 2015-2016 Yusuke Suzuki <yusuke.suzuki@sslab.ics.keio.ac.jp>

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
  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAfree AND
  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
  THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#include "device_context.cuh"
#include "device_loop.cuh"
#include "dump_memory.cuh"
#include "function.cuh"
#include "memcpy_io.cuh"
#include "sync_read_write.h"
#include "utility.h"
namespace gloop {

// Initialize a device loop per thread block.
__device__ __shared__ DeviceLoop sharedDeviceLoop;

__device__ void DeviceLoop::initializeImpl(volatile uint32_t* signal, DeviceContext deviceContext)
{
    GLOOP_ASSERT_SINGLE_THREAD();

    m_deviceContext = deviceContext;
    m_channels = deviceContext.channels + (GLOOP_BID() * GLOOP_SHARED_SLOT_SIZE);
    m_slots = reinterpret_cast<DeviceCallback*>(&context()->slots);
    m_signal = signal;

#if defined(GLOOP_ENABLE_HIERARCHICAL_SLOT_MEMORY)
    m_scratchIndex1 = invalidPosition();
    m_scratchIndex2 = invalidPosition();
#endif
}

__device__ void DeviceLoop::initialize(volatile uint32_t* signal, DeviceContext deviceContext)
{
    initializeImpl(signal, deviceContext);
    m_control.initialize(deviceContext.logicalBlocks);
}

__device__ void DeviceLoop::initialize(volatile uint32_t* signal, DeviceContext deviceContext, ResumeTag)
{
    initializeImpl(signal, deviceContext);
    resume();
}

__device__ void DeviceLoop::suspend()
{
    // FIXME: always save.
    GLOOP_ASSERT_SINGLE_THREAD();
    // __threadfence_system();  // FIXME
    DeviceContext::PerBlockContext* blockContext = context();
    blockContext->control = m_control;
    if (m_control.freeSlots != DeviceContext::DeviceLoopControl::allFilledFreeSlots()) {
        atomicAdd(m_deviceContext.pending, 1);
    }
#if defined(GLOOP_ENABLE_HIERARCHICAL_SLOT_MEMORY)
    if (m_scratchIndex1 != invalidPosition()) {
        new (reinterpret_cast<DeviceCallback*>(&blockContext->slots) + m_scratchIndex1) DeviceCallback(*reinterpret_cast<DeviceCallback*>(&m_scratch1));
        reinterpret_cast<DeviceCallback*>(&m_scratch1)->~DeviceCallback();
    }
    if (m_scratchIndex2 != invalidPosition()) {
        new (reinterpret_cast<DeviceCallback*>(&blockContext->slots) + m_scratchIndex2) DeviceCallback(*reinterpret_cast<DeviceCallback*>(&m_scratch2));
        reinterpret_cast<DeviceCallback*>(&m_scratch2)->~DeviceCallback();
    }
#endif
    __threadfence_system();  // FIXME
}

__device__ void DeviceLoop::resume()
{
    GLOOP_ASSERT_SINGLE_THREAD();
    // __threadfence_system();  // FIXME
    DeviceContext::PerBlockContext* blockContext = context();
    m_control = blockContext->control;
    // __threadfence_system();  // FIXME
}

__device__ void DeviceLoop::freeOnePage(void* aPage)
{
    GLOOP_ASSERT_SINGLE_THREAD();
    uint32_t pos = position(static_cast<DeviceContext::OnePage*>(aPage));
    m_control.freePages |= (1UL << pos);
    GPU_ASSERT(pos < GLOOP_SHARED_PAGE_COUNT);
    int freePageWaitingCallbackPlusOne = __ffs(m_control.pageSleepSlots);
    if (freePageWaitingCallbackPlusOne) {
        m_control.wakeupSlots |= (1U << (freePageWaitingCallbackPlusOne - 1));
    }
}

}  // namespace gloop
