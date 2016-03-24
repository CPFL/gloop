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
#ifndef GLOOP_DEVICE_LOOP_INLINE_CU_H_
#define GLOOP_DEVICE_LOOP_INLINE_CU_H_
#include "device_loop.cuh"
namespace gloop {

__device__ uint32_t DeviceLoop::position(DeviceCallback* callback)
{
    GLOOP_ASSERT_SINGLE_THREAD();
    return callback - m_slots;
}

__device__ uint32_t DeviceLoop::position(IPC* ipc)
{
    GLOOP_ASSERT_SINGLE_THREAD();
    return ipc - channel();
}

__device__ uint32_t DeviceLoop::position(DeviceContext::OnePage* page)
{
    GLOOP_ASSERT_SINGLE_THREAD();
    return page - pages();
}

__device__ auto DeviceLoop::channel() const -> IPC*
{
    GLOOP_ASSERT_SINGLE_THREAD();
    return m_deviceContext.channels + (GLOOP_BID() * GLOOP_SHARED_SLOT_SIZE);
}

__device__ auto DeviceLoop::context() const -> DeviceContext::PerBlockContext*
{
    GLOOP_ASSERT_SINGLE_THREAD();
    return m_deviceContext.context + GLOOP_BID();
}

__device__ auto DeviceLoop::pages() const -> DeviceContext::OnePage*
{
    GLOOP_ASSERT_SINGLE_THREAD();
    return m_deviceContext.pages + (GLOOP_BID() * GLOOP_SHARED_PAGE_COUNT);
}

template<typename Lambda>
__device__ void DeviceLoop::allocOnePage(Lambda lambda)
{
    __shared__ void* page;
    BEGIN_SINGLE_THREAD
    {
        page = nullptr;
        int freePagePosPlusOne = __ffs(m_control.freePages);
        if (freePagePosPlusOne == 0) {
            uint32_t pos = enqueueSleep([lambda](DeviceLoop* loop, volatile request::Request* req) {
                loop->allocOnePage(lambda);
            });
            m_control.pageSleepSlots |= (1ULL << pos);
        } else {
            int pagePos = freePagePosPlusOne - 1;
            page = pages() + pagePos;
            m_control.freePages &= ~(1UL << pagePos);
        }
    }
    END_SINGLE_THREAD
    if (!page)
        return;
    lambda(this, page);
}

template<typename Lambda>
__device__ void DeviceLoop::enqueueLater(Lambda lambda)
{
    GLOOP_ASSERT_SINGLE_THREAD();
    uint32_t pos = enqueueSleep(lambda);
    m_control.wakeupSlots |= (1ULL << pos);
}

template<typename Lambda>
__device__ uint32_t DeviceLoop::enqueueSleep(Lambda lambda)
{
    GLOOP_ASSERT_SINGLE_THREAD();
    uint32_t pos = allocate(lambda);
    m_control.sleepSlots |= (1ULL << pos);
    return pos;
}

template<typename Lambda>
__device__ IPC* DeviceLoop::enqueueIPC(Lambda lambda)
{
    GLOOP_ASSERT_SINGLE_THREAD();
    uint32_t pos = allocate(lambda);
    IPC* result = channel() + pos;
    GPU_ASSERT(pos < GLOOP_SHARED_SLOT_SIZE);
    return result;
}

template<typename Lambda>
__device__ uint32_t DeviceLoop::allocate(Lambda lambda)
{
    GLOOP_ASSERT_SINGLE_THREAD();
    int pos = __ffsll(m_control.freeSlots) - 1;
    GPU_ASSERT(pos >= 0 && pos <= GLOOP_SHARED_SLOT_SIZE);
    GPU_ASSERT(m_control.freeSlots & (1ULL << pos));
    m_control.freeSlots &= ~(1ULL << pos);

    new (m_slots + pos) DeviceCallback(lambda);
    m_control.pending += 1;

    return pos;
}

}  // namespace gloop
#endif  // GLOOP_DEVICE_LOOP_INLINE_CU_H_
