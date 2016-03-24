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

__device__ DeviceLoop::DeviceLoop(volatile uint32_t* signal, DeviceContext deviceContext, size_t size)
    : m_deviceContext(deviceContext)
    , m_slots(reinterpret_cast<DeviceCallback*>(&context()->slots))
    , m_control()
    , m_signal(signal)
{
    GLOOP_ASSERT_SINGLE_THREAD();
    GPU_ASSERT(size >= GLOOP_SHARED_SLOT_SIZE);
}

__device__ auto DeviceLoop::dequeue() -> uint32_t
{
    GLOOP_ASSERT_SINGLE_THREAD();
    // __threadfence_system();
    bool shouldExit = false;
    for (uint32_t i = 0; i < GLOOP_SHARED_SLOT_SIZE; ++i) {
        // Look into ICP status to run callbacks.
        uint64_t bit = 1ULL << i;
        if (!(m_control.freeSlots & bit)) {
            if (m_control.sleepSlots & bit) {
                if (m_control.wakeupSlots & bit) {
                    m_control.sleepSlots &= ~bit;
                    m_control.wakeupSlots &= ~bit;
                    return i;
                }
            } else {
                IPC* ipc = channel() + i;
                Code code = ipc->peek();
                if (code == Code::Complete) {
                    ipc->emit(Code::None);
                    GPU_ASSERT(ipc->peek() != Code::Complete);
                    GPU_ASSERT(ipc->peek() == Code::None);
                    return i;
                }

                // FIXME: More careful exit routine.
                // Let's exit to meet ExitRequired requirements.
                if (code == Code::ExitRequired) {
                    shouldExit = true;
                }
            }
        }
    }

    if (shouldExit) {
        return shouldExitPosition();
    }
    return invalidPosition();
}

__device__ void DeviceLoop::drain()
{
    __shared__ uint64_t start;
    __shared__ uint32_t position;
    __shared__ DeviceCallback* callback;
    __shared__ IPC* ipc;

    BEGIN_SINGLE_THREAD
    {
        start = clock64();
        if (m_control.pending) {
            position = dequeue();
            if (isValidPosition(position)) {
                callback = slots(position);
                ipc = channel() + position;
            } else {
                callback = nullptr;
            }
        } else {
            position = shouldExitPosition();
        }
    }
    END_SINGLE_THREAD

    while (position != shouldExitPosition()) {
        if (callback) {
            // __threadfence_system();  // IPC and Callback.
            // __threadfence_block();
            __syncthreads();  // FIXME

            // One shot function always destroys the function and syncs threads.
            (*callback)(this, ipc->request());

            // __syncthreads();  // FIXME
            // __threadfence_block();
        }

        // __threadfence_block();
        BEGIN_SINGLE_THREAD
        {
            if (callback) {
                deallocate(callback, position);
            }

            uint64_t now = clock64();
            if ((now - start) > m_deviceContext.killClock) {
                if (gloop::readNoCache<uint32_t>(m_signal) != 0) {
                    position = shouldExitPosition();
                }
                start = now;
            }

            if (position != shouldExitPosition()) {
                if (m_control.pending) {
                    position = dequeue();
                    if (isValidPosition(position)) {
                        callback = slots(position);
                        ipc = channel() + position;
                    } else {
                        callback = nullptr;
                    }
                } else {
                    position = shouldExitPosition();
                }
            }
        }
        END_SINGLE_THREAD
    }
    // __threadfence_block();
    BEGIN_SINGLE_THREAD
    {
        suspend();
    }
    END_SINGLE_THREAD
    // __threadfence_block();
}

__device__ void DeviceLoop::deallocate(DeviceCallback* callback, uint32_t pos)
{
    GLOOP_ASSERT_SINGLE_THREAD();
    // printf("pos:(%u)\n", (unsigned)pos);
    GPU_ASSERT(pos >= 0 && pos <= GLOOP_SHARED_SLOT_SIZE);
    GPU_ASSERT(!(m_control.freeSlots & (1ULL << pos)));

    // We are using one shot function. After calling the function, destruction is already done.
    // callback->~DeviceCallback();
    if (pos == m_scratchIndex1) {
        m_scratchIndex1 = invalidPosition();
    } else if (pos == m_scratchIndex2) {
        m_scratchIndex2 = invalidPosition();
    }

    m_control.freeSlots |= (1ULL << pos);
    m_control.pending -= 1;
}

__device__ void DeviceLoop::suspend()
{
    // FIXME: always save.
    GLOOP_ASSERT_SINGLE_THREAD();
    // __threadfence_system();  // FIXME
    DeviceContext::PerBlockContext* blockContext = context();
    blockContext->control = m_control;
    if (m_control.pending) {
        atomicAdd(m_deviceContext.pending, 1);
    }
    if (m_scratchIndex1 != invalidPosition()) {
        new (reinterpret_cast<DeviceCallback*>(&blockContext->slots) + m_scratchIndex1) DeviceCallback(*reinterpret_cast<DeviceCallback*>(&m_scratch1));
        reinterpret_cast<DeviceCallback*>(&m_scratch1)->~DeviceCallback();
    }
    if (m_scratchIndex2 != invalidPosition()) {
        new (reinterpret_cast<DeviceCallback*>(&blockContext->slots) + m_scratchIndex2) DeviceCallback(*reinterpret_cast<DeviceCallback*>(&m_scratch2));
        reinterpret_cast<DeviceCallback*>(&m_scratch2)->~DeviceCallback();
    }
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
    uint64_t pos = position(static_cast<DeviceContext::OnePage*>(aPage));
    m_control.freePages |= (1UL << pos);
    GPU_ASSERT(pos < GLOOP_SHARED_PAGE_COUNT);
    int freePageWaitingCallbackPlusOne = __ffsll(m_control.pageSleepSlots);
    if (freePageWaitingCallbackPlusOne) {
        m_control.wakeupSlots |= (1ULL << (freePageWaitingCallbackPlusOne - 1));
    }
}

}  // namespace gloop
