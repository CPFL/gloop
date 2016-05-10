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
#ifndef GLOOP_DEVICE_LOOP_INLINES_CU_H_
#define GLOOP_DEVICE_LOOP_INLINES_CU_H_
namespace gloop {

GLOOP_ALWAYS_INLINE __device__ void IPC::emit(DeviceLoop* loop, Code code)
{
    syncWrite(&loop->m_codes[position], static_cast<int32_t>(code));
}

GLOOP_ALWAYS_INLINE __device__ Code IPC::peek(DeviceLoop* loop)
{
    return readNoCache<Code>(&loop->m_codes[position]);
}

GLOOP_ALWAYS_INLINE __device__ request::Payload* IPC::request(DeviceLoop* loop)
{
    return &loop->m_payloads[position];
}

__device__ uint32_t DeviceLoop::position(DeviceContext::OnePage* page)
{
    GLOOP_ASSERT_SINGLE_THREAD();
    return page - pages();
}

__device__ bool DeviceLoop::isValidPosition(uint32_t position)
{
    return position < GLOOP_SHARED_SLOT_SIZE;
}

__device__ auto DeviceLoop::slots(uint32_t position) -> DeviceCallback*
{
    GLOOP_ASSERT_SINGLE_THREAD();
#if defined(GLOOP_ENABLE_HIERARCHICAL_SLOT_MEMORY)
    if (position == m_scratchIndex1) {
        return reinterpret_cast<DeviceCallback*>(&m_scratch1);
    }
    if (position == m_scratchIndex2) {
        return reinterpret_cast<DeviceCallback*>(&m_scratch2);
    }
#endif
    return m_slots + position;
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
            m_control.pageSleepSlots |= (1U << pos);
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
    m_control.wakeupSlots |= (1U << pos);
}

template<typename Lambda>
__device__ uint32_t DeviceLoop::enqueueSleep(Lambda lambda)
{
    GLOOP_ASSERT_SINGLE_THREAD();
    uint32_t pos = allocate(lambda);
    m_control.sleepSlots |= (1U << pos);
    m_control.wakeupSlots &= ~(1U << pos);
    return pos;
}

template<typename Lambda>
__device__ IPC DeviceLoop::enqueueIPC(Lambda lambda)
{
    GLOOP_ASSERT_SINGLE_THREAD();
    uint32_t pos = allocate(lambda);
    GPU_ASSERT(pos < GLOOP_SHARED_SLOT_SIZE);
    return { pos };
}

template<typename Lambda>
__device__ uint32_t DeviceLoop::allocate(Lambda lambda)
{
    GLOOP_ASSERT_SINGLE_THREAD();
    int pos = __ffs(m_control.freeSlots) - 1;
    GPU_ASSERT(pos >= 0 && pos <= GLOOP_SHARED_SLOT_SIZE);
    GPU_ASSERT(m_control.freeSlots & (1U << pos));
    m_control.freeSlots &= ~(1U << pos);

    void* target = m_slots + pos;
#if defined(GLOOP_ENABLE_HIERARCHICAL_SLOT_MEMORY)
    if (m_scratchIndex1 == invalidPosition()) {
        m_scratchIndex1 = pos;
        target = &m_scratch1;
    } else if (m_scratchIndex2 == invalidPosition()) {
        m_scratchIndex2 = pos;
        target = &m_scratch2;
    }
#endif

    new (target) DeviceCallback(lambda);

    return pos;
}

__device__ auto DeviceLoop::dequeue() -> uint32_t
{
    GLOOP_ASSERT_SINGLE_THREAD();

    uint32_t freeSlots = m_control.freeSlots;
    if (freeSlots == DeviceContext::DeviceLoopControl::allFilledFreeSlots()) {
        return shouldExitPosition();
    }

    // __threadfence_system();
    // We first search wake up slots. It is always ready to execute.
    // And we can get the slot without costly DMA.
    uint32_t allocatedSlots = freeSlots ^ DeviceContext::DeviceLoopControl::allFilledFreeSlots();
    uint32_t wakeupSlots = (allocatedSlots & m_control.sleepSlots) & m_control.wakeupSlots;

    if (wakeupSlots) {
        int position = __ffs(wakeupSlots) - 1;
        m_control.sleepSlots &= ~(1U << position);
        return position;
    }

    allocatedSlots &= ~m_control.sleepSlots;

    bool shouldExit = false;
    for (uint32_t i = 0; i < GLOOP_SHARED_SLOT_SIZE; ++i) {
        // Look into ICP status to run callbacks.
        uint32_t bit = 1U << i;
        if (allocatedSlots & bit) {
            IPC ipc { i };
            Code code = ipc.peek(this);
            if (code == Code::Complete) {
                ipc.emit(this, Code::None);
                GPU_ASSERT(ipc.peek(this) != Code::Complete);
                GPU_ASSERT(ipc.peek(this) == Code::None);
                return i;
            }

            // FIXME: More careful exit routine.
            // Let's exit to meet ExitRequired requirements.
            if (code == Code::ExitRequired) {
                shouldExit = true;
            }
        }
    }

    if (shouldExit) {
        return shouldExitPosition();
    }
    return invalidPosition();
}

__device__ void DeviceLoop::deallocate(uint32_t pos)
{
    GLOOP_ASSERT_SINGLE_THREAD();
    // printf("pos:(%u)\n", (unsigned)pos);
    GPU_ASSERT(pos >= 0 && pos <= GLOOP_SHARED_SLOT_SIZE);
    GPU_ASSERT(!(m_control.freeSlots & (1U << pos)));

    // We are using one shot function. After calling the function, destruction is already done.
    // callback->~DeviceCallback();
#if defined(GLOOP_ENABLE_HIERARCHICAL_SLOT_MEMORY)
    if (pos == m_scratchIndex1) {
        m_scratchIndex1 = invalidPosition();
    } else if (pos == m_scratchIndex2) {
        m_scratchIndex2 = invalidPosition();
    }
#endif

    m_control.freeSlots |= (1U << pos);
}

__device__ int DeviceLoop::drain()
{
    uint64_t start;
    uint64_t killClock;
    __shared__ uint32_t position;
    __shared__ DeviceCallback* callback;

    BEGIN_SINGLE_THREAD
    {
        killClock = m_deviceContext.killClock;
        start = m_start;
        callback = nullptr;
        position = invalidPosition();
    }
    END_SINGLE_THREAD

    while (position != shouldExitPosition()) {
        if (callback) {
            // __threadfence_system();  // IPC and Callback.
            // __threadfence_block();
            // __syncthreads();  // FIXME

            // printf("%llu %u\n", (unsigned long long)(clock64() - start), (unsigned)position);
            // One shot function always destroys the function and syncs threads.
            (*callback)(this, &m_payloads[position]);

            // __syncthreads();  // FIXME
            // __threadfence_block();
        }

        // __threadfence_block();
        BEGIN_SINGLE_THREAD
        {
            // 100 - 130 clock
            if (callback) {
                deallocate(position);
            }

#if 1
            // 100 clock

            uint64_t now = clock64();
            if (((now - start) > killClock)) {
                start = ((now - start) / killClock * killClock);
                if (gloop::readNoCache<uint32_t>(m_signal) != 0) {
                    position = shouldExitPosition();
                    goto next;
                }
            }
#endif

            // 200 clock.
            callback = nullptr;
            position = dequeue();
            if (isValidPosition(position)) {
                callback = slots(position);
            }
next:
        }
        END_SINGLE_THREAD
    }
    // __threadfence_block();
    __shared__ int suspended;
    BEGIN_SINGLE_THREAD
    {
        suspended = suspend();
    }
    END_SINGLE_THREAD
    // __threadfence_block();
    return suspended;
}

}  // namespace gloop
#endif  // GLOOP_DEVICE_LOOP_INLINES_CU_H_
