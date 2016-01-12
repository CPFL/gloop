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
#include "device_loop.cuh"
#include "function.cuh"
namespace gloop {

__device__ DeviceLoop::DeviceLoop(Callback* buffer, size_t size)
    : m_slots(buffer)
    , m_put(0)
    , m_get(0)
    , m_used(static_cast<decltype(m_used)>(-1))
{
    GPU_ASSERT(size >= GLOOP_SHARED_SLOT_SIZE);
}

static inline __device__ void copyCallback(const DeviceLoop::Callback* src, DeviceLoop::Callback* dst)
{
    uint64_t* pointer = (uint64_t*)(src);
    uint64_t* put = (uint64_t*)(dst);
    static_assert(sizeof(DeviceLoop::Callback) % sizeof(uint64_t) == 0, "Callback size should be n * sizeof(uint64_t)");
    for (size_t i = 0; i < (sizeof(DeviceLoop::Callback) / sizeof(uint64_t)); ++i) {
        *put++ = *pointer++;
    }
}

__device__ void DeviceLoop::enqueue(Callback lambda)
{
    static_assert(sizeof(Callback) % sizeof(uint64_t) == 0, "OK.");
    BEGIN_SINGLE_THREAD
    {
        uint32_t position = allocate();
        m_queue[m_put++ % GLOOP_SHARED_SLOT_SIZE] = position;
        GPU_ASSERT(m_put != m_get);
        GPU_ASSERT(position < GLOOP_SHARED_SLOT_SIZE);
        copyCallback(&lambda, m_slots + position);
    }
    END_SINGLE_THREAD
}

__device__ auto DeviceLoop::dequeue() -> Callback*
{
    __shared__ Callback* result;
    BEGIN_SINGLE_THREAD
    {
        GPU_ASSERT(m_get != m_put);
        uint32_t position = m_queue[m_get++ % GLOOP_SHARED_SLOT_SIZE];
        result = m_slots + position;
    }
    END_SINGLE_THREAD
    return result;
}

__device__ bool DeviceLoop::drain()
{
    while (!done()) {
        Callback* callback = dequeue();
        (*callback)(this, 0);
        deallocate(callback);
    }
    return true;
}

__device__ uint32_t DeviceLoop::allocate()
{
    GLOOP_ASSERT_SINGLE_THREAD();
    int position = __ffsll(m_used) - 1;
    GPU_ASSERT(position >= 0);
    GPU_ASSERT(m_used & (1ULL << position));
    m_used &= ~(1ULL << position);
    return position;
}

__device__ void DeviceLoop::deallocate(Callback* callback)
{
    BEGIN_SINGLE_THREAD
    {
        uint32_t position = (callback - m_slots);
        GPU_ASSERT(!(m_used & (1ULL << position)));
        m_used |= (1ULL << position);
    }
    END_SINGLE_THREAD
}

}  // namespace gloop
