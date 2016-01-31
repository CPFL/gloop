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
#include "function.cuh"
namespace gloop {

__device__ DeviceLoop::DeviceLoop(DeviceContext deviceContext, UninitializedStorage* buffer, size_t size)
    : m_deviceContext(deviceContext)
    , m_slots(reinterpret_cast<Callback*>(buffer))
    , m_control()
{
    GPU_ASSERT(size >= GLOOP_SHARED_SLOT_SIZE);
}

// TODO: Callback should be treated as destructible.
static inline __device__ void copyCallback(const DeviceLoop::Callback* src, DeviceLoop::Callback* dst)
{
    uint64_t* pointer = (uint64_t*)(src);
    uint64_t* put = (uint64_t*)(dst);
    static_assert(sizeof(DeviceLoop::Callback) % sizeof(uint64_t) == 0, "Callback size should be n * sizeof(uint64_t)");
    for (size_t i = 0; i < (sizeof(DeviceLoop::Callback) / sizeof(uint64_t)); ++i) {
        *put++ = *pointer++;
    }
}

__device__ IPC* DeviceLoop::enqueueIPC(Callback lambda)
{
    __shared__ IPC* result;
    BEGIN_SINGLE_THREAD
    {
        uint32_t pos = allocate(&lambda);
        m_control.queue[m_control.put++ % GLOOP_SHARED_SLOT_SIZE] = pos;
        result = channel() + pos;
        GPU_ASSERT(m_control.put != m_control.get);
        GPU_ASSERT(pos < GLOOP_SHARED_SLOT_SIZE);
    }
    END_SINGLE_THREAD
    return result;
}

__device__ void DeviceLoop::enqueueLater(Callback lambda)
{
    BEGIN_SINGLE_THREAD
    {
        uint32_t pos = enqueueSleep(lambda);
        m_control.wakeup |= (1ULL << pos);
    }
    END_SINGLE_THREAD
}

__device__ auto DeviceLoop::dequeue() -> Callback*
{
    __shared__ Callback* result;
    BEGIN_SINGLE_THREAD
    {
        result = nullptr;
        for (uint32_t i = 0; i < GLOOP_SHARED_SLOT_SIZE; ++i) {
            // Look into ICP status to run callbacks.
            uint64_t bit = 1ULL << i;
            if (m_control.sleep & bit) {
                if (m_control.wakeup & bit) {
                    m_control.sleep &= ~bit;
                    m_control.wakeup &= ~bit;
                    result = m_slots + i;
                    break;
                }
            } else {
                if (!(m_control.free & bit)) {
                    IPC* ipc = channel() + i;
                    if (ipc->peek() == Code::Complete) {
                        ipc->emit(Code::None);
                        GPU_ASSERT(ipc->peek() != Code::Complete);
                        result = m_slots + i;
                        break;
                    }
                }
            }
        }
    }
    END_SINGLE_THREAD
    return result;
}

__device__ bool DeviceLoop::drain()
{
    while (true) {
        if (!m_control.pending) {
            break;
        }

        if (Callback* callback = dequeue()) {
            uint32_t pos = position(callback);
            GPU_ASSERT(pos >= 0 && pos <= GLOOP_SHARED_SLOT_SIZE);
            IPC* ipc = channel() + pos;
            GPU_ASSERT(ipc->peek() == Code::None);
            (*callback)(this, ipc->request());
            deallocate(callback);
        }
        // FIXME
        break;
    }
    if (m_control.pending) {
        // Flush pending jobs to global pending status.
        suspend();
    }
    return true;
}

__device__ uint32_t DeviceLoop::allocate(const Callback* lambda)
{
    GLOOP_ASSERT_SINGLE_THREAD();
    int pos = __ffsll(m_control.free) - 1;
    GPU_ASSERT(pos >= 0 && pos <= GLOOP_SHARED_SLOT_SIZE);
    GPU_ASSERT(m_control.free & (1ULL << pos));
    m_control.free &= ~(1ULL << pos);
    copyCallback(lambda, m_slots + pos);
    m_control.pending += 1;
    return pos;
}

__device__ void DeviceLoop::deallocate(Callback* callback)
{
    BEGIN_SINGLE_THREAD
    {
        uint32_t pos = position(callback);
        GPU_ASSERT(pos >= 0 && pos <= GLOOP_SHARED_SLOT_SIZE);
        GPU_ASSERT(!(m_control.free & (1ULL << pos)));
        m_control.free |= (1ULL << pos);
        m_control.pending -= 1;
    }
    END_SINGLE_THREAD
}

static GLOOP_ALWAYS_INLINE __device__ void copyContext(void* src, void* dst)
{
    typedef DeviceLoop::UninitializedStorage UninitializedStorage;
    UninitializedStorage* storage = reinterpret_cast<UninitializedStorage*>(dst);
    if (GLOOP_TMAX() >= GLOOP_SHARED_SLOT_SIZE) {
        int target = GLOOP_TID();
        if (target < GLOOP_SHARED_SLOT_SIZE) {
            *(storage + target) = *(reinterpret_cast<UninitializedStorage*>(src) + target);
        }
    } else {
        int count = GLOOP_SHARED_SLOT_SIZE / GLOOP_TMAX();
        if (GLOOP_SHARED_SLOT_SIZE % GLOOP_TMAX()) {
            ++count;
        }
        for (int i = 0; i < count; ++i) {
            int target = i * GLOOP_TMAX() + GLOOP_TID();
            if (target < GLOOP_SHARED_SLOT_SIZE) {
                *(storage + target) = *(reinterpret_cast<UninitializedStorage*>(src) + target);
            }
        }
    }
}

__device__ void DeviceLoop::suspend()
{
    PerBlockContext* blockContext = context();
    copyContext(m_slots, &blockContext->slots);
    BEGIN_SINGLE_THREAD
    {
        blockContext->control = m_control;
        atomicAdd(m_deviceContext.pending, 1);
    }
    END_SINGLE_THREAD
}

__device__ void DeviceLoop::resume()
{
    PerBlockContext* blockContext = context();
    copyContext(&blockContext->slots, m_slots);
    BEGIN_SINGLE_THREAD
    {
        m_control = blockContext->control;
    }
    END_SINGLE_THREAD
}

__device__ void DeviceLoop::emit(Code code, IPC* ipc)
{
    BEGIN_SINGLE_THREAD
    {
        ipc->emit(code);
    }
    END_SINGLE_THREAD
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
    BEGIN_SINGLE_THREAD
    {
        uint32_t pos = position(static_cast<OnePage*>(aPage));
        m_control.freePages |= (1ULL << pos);
        GPU_ASSERT(pos < GLOOP_SHARED_PAGE_COUNT);
        int freePageWaitingCallbackPlusOne = __ffsll(m_control.m_pageSleep);
        if (freePageWaitingCallbackPlusOne) {
            m_control.wakeup |= (1ULL << (freePageWaitingCallbackPlusOne - 1));
        }
    }
    END_SINGLE_THREAD
}

}  // namespace gloop
