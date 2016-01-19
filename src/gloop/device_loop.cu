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
  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
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

__device__ request::Request* DeviceLoop::enqueue(Callback lambda)
{
    __shared__ request::Request* result;
    static_assert(sizeof(Callback) % sizeof(uint64_t) == 0, "OK.");
    BEGIN_SINGLE_THREAD
    {
        uint32_t pos = allocate();
        m_control.queue[m_control.put++ % GLOOP_SHARED_SLOT_SIZE] = pos;
        result = channel()->requests + pos;
        GPU_ASSERT(m_control.put != m_control.get);
        GPU_ASSERT(pos < GLOOP_SHARED_SLOT_SIZE);
        copyCallback(&lambda, m_slots + pos);
    }
    END_SINGLE_THREAD
    return result;
}

__device__ auto DeviceLoop::dequeue() -> Callback*
{
    __shared__ Callback* result;
    BEGIN_SINGLE_THREAD
    {
        if (m_control.put == m_control.get) {
            result = nullptr;
        } else {
            GPU_ASSERT(m_control.get != m_control.put);
            uint32_t pos = m_control.queue[m_control.get++ % GLOOP_SHARED_SLOT_SIZE];
            result = m_slots + pos;
        }
    }
    END_SINGLE_THREAD
    return result;
}

__device__ bool DeviceLoop::drain()
{
    while (true) {
        // FIXME: more coarse grained checking.
        if (g_channel->peek()) {
            suspend();
            return false;
        }

        if (Callback* callback = dequeue()) {
            uint32_t pos = position(callback);
            (*callback)(this, &channel()->requests[pos]);
            deallocate(callback);
        } else {
            break;
        }
    }
    return true;
}

__device__ uint32_t DeviceLoop::allocate()
{
    GLOOP_ASSERT_SINGLE_THREAD();
    int pos = __ffsll(m_control.used) - 1;
    GPU_ASSERT(pos >= 0);
    GPU_ASSERT(m_control.used & (1ULL << pos));
    m_control.used &= ~(1ULL << pos);
    return pos;
}

__device__ void DeviceLoop::deallocate(Callback* callback)
{
    BEGIN_SINGLE_THREAD
    {
        uint32_t pos = position(callback);
        GPU_ASSERT(!(m_control.used & (1ULL << pos)));
        m_control.used |= (1ULL << pos);
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

__device__ void DeviceLoop::emit(request::Request* request)
{
    BEGIN_SINGLE_THREAD
    {
        uint32_t pos = position(request);
    }
    END_SINGLE_THREAD
}

}  // namespace gloop
