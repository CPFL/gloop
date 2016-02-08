/*
  Copyright (C) 2015 Yusuke Suzuki <yusuke.suzuki@sslab.ics.keio.ac.jp>

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
#include <gpufs/libgpufs/fs_calls.cu.h>
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
    , m_slots(reinterpret_cast<Callback*>(&context()->slots))
    , m_control()
    , m_signal(signal)
{
    GLOOP_ASSERT_SINGLE_THREAD();
    GPU_ASSERT(size >= GLOOP_SHARED_SLOT_SIZE);
}

__device__ IPC* DeviceLoop::enqueueIPC(Callback lambda)
{
    GLOOP_ASSERT_SINGLE_THREAD();
    uint32_t pos = allocate(&lambda);
    m_control.queue[m_control.put++ % GLOOP_SHARED_SLOT_SIZE] = pos;
    IPC* result = channel() + pos;
    GPU_ASSERT(m_control.put != m_control.get);
    GPU_ASSERT(pos < GLOOP_SHARED_SLOT_SIZE);
    return result;
}

__device__ void DeviceLoop::enqueueLater(Callback lambda)
{
    GLOOP_ASSERT_SINGLE_THREAD();
    uint32_t pos = enqueueSleep(lambda);
    m_control.wakeup |= (1ULL << pos);
}

__device__ auto DeviceLoop::dequeue(bool& shouldExit) -> Callback*
{
    GLOOP_ASSERT_SINGLE_THREAD();
    __threadfence_system();
    for (uint32_t i = 0; i < GLOOP_SHARED_SLOT_SIZE; ++i) {
        // Look into ICP status to run callbacks.
        uint64_t bit = 1ULL << i;
        if (m_control.sleep & bit) {
            if (m_control.wakeup & bit) {
                m_control.sleep &= ~bit;
                m_control.wakeup &= ~bit;
                return m_slots + i;
            }
        } else if (!(m_control.free & bit)) {
            IPC* ipc = channel() + i;
            Code code = ipc->peek();
            if (code == Code::Complete) {
                ipc->emit(Code::None);
                GPU_ASSERT(ipc->peek() != Code::Complete);
                GPU_ASSERT(ipc->peek() == Code::None);
                return m_slots + i;
            }

            // FIXME: More careful exit decision.
            if (code == Code::ExitRequired) {
                shouldExit = true;
            }
        }
    }
    return nullptr;
}

__device__ void DeviceLoop::drain()
{
    while (true) {
        __shared__ uint32_t pending;
        __shared__ Callback* callback;
        __shared__ IPC* ipc;
        BEGIN_SINGLE_THREAD
        {
            pending = m_control.pending;

            if (pending) {
                bool shouldExit = false;
                callback = dequeue(shouldExit);
                if (callback) {
                    uint32_t pos = position(callback);
                    ipc = channel() + pos;
                } else {
                    // FIXME: More careful exit routine.
                    // Let's exit to meet ExitRequired requirements.
                    if (shouldExit) {
                        pending = 0;
                    }
                }
            }
        }
        END_SINGLE_THREAD
        __threadfence_block();

        if (!pending) {
            break;
        }

        if (callback) {
            __threadfence_system();  // IPC and Callback.
            __syncthreads();  // FIXME
            __threadfence_block();
            (*callback)(this, ipc->request());
            __syncthreads();  // FIXME
            __threadfence_block();
            deallocate(callback);
        }

        __shared__ bool signaled;
        __threadfence_block();
        BEGIN_SINGLE_THREAD
        {
            // FIXME: Sometimes, we should execute this. Taking tick in GPU kernel is nice.
            signaled = gloop::readNoCache<uint32_t>(m_signal) != 0;
        }
        END_SINGLE_THREAD
        __threadfence_block();
        if (signaled) {
            break;
        }
        // break;
    }
    __threadfence_block();
    BEGIN_SINGLE_THREAD
    {
        suspend();
    }
    END_SINGLE_THREAD
    __threadfence_block();
}

__device__ uint32_t DeviceLoop::allocate(const Callback* lambda)
{
    GLOOP_ASSERT_SINGLE_THREAD();
    int pos = __ffsll(m_control.free) - 1;
    GPU_ASSERT(pos >= 0 && pos <= GLOOP_SHARED_SLOT_SIZE);
    GPU_ASSERT(m_control.free & (1ULL << pos));
    m_control.free &= ~(1ULL << pos);

    new (m_slots + pos) Callback(*lambda);
    m_control.pending += 1;

    return pos;
}

__device__ void DeviceLoop::deallocate(Callback* callback)
{
    BEGIN_SINGLE_THREAD
    {
        uint32_t pos = position(callback);
        // printf("pos:(%u)\n", (unsigned)pos);
        GPU_ASSERT(pos >= 0 && pos <= GLOOP_SHARED_SLOT_SIZE);
        GPU_ASSERT(!(m_control.free & (1ULL << pos)));

        callback->~Callback();

        m_control.free |= (1ULL << pos);
        m_control.pending -= 1;
    }
    END_SINGLE_THREAD
}

__device__ void DeviceLoop::suspend()
{
    // FIXME: always save.
    GLOOP_ASSERT_SINGLE_THREAD();
    __threadfence_system();  // FIXME
    PerBlockContext* blockContext = context();
    blockContext->control = m_control;
    if (m_control.pending) {
        atomicAdd(m_deviceContext.pending, 1);
    }
    __threadfence_system();  // FIXME
}

__device__ void DeviceLoop::resume()
{
    GLOOP_ASSERT_SINGLE_THREAD();
    __threadfence_system();  // FIXME
    PerBlockContext* blockContext = context();
    m_control = blockContext->control;
    __threadfence_system();  // FIXME
}

__device__ uint32_t DeviceLoop::enqueueSleep(const Callback& lambda)
{
    GLOOP_ASSERT_SINGLE_THREAD();
    uint32_t pos = allocate(&lambda);
    m_control.queue[m_control.put++ % GLOOP_SHARED_SLOT_SIZE] = pos;
    m_control.sleep |= (1ULL << pos);
    return pos;
}

__device__ void DeviceLoop::freeOnePage(void* aPage)
{
    GLOOP_ASSERT_SINGLE_THREAD();
    uint32_t pos = position(static_cast<OnePage*>(aPage));
    m_control.freePages |= (1ULL << pos);
    GPU_ASSERT(pos < GLOOP_SHARED_PAGE_COUNT);
    int freePageWaitingCallbackPlusOne = __ffsll(m_control.m_pageSleep);
    if (freePageWaitingCallbackPlusOne) {
        m_control.wakeup |= (1ULL << (freePageWaitingCallbackPlusOne - 1));
    }
}

}  // namespace gloop
