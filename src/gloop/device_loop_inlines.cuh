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

#pragma once

#include "config.h"
#include "device_context.cuh"
#include "device_loop.cuh"
#include "function.cuh"
#include "sync_read_write.h"
#include "utility.h"
#include <utility>
namespace gloop {

template <typename DeviceLoop>
GLOOP_ALWAYS_INLINE __device__ void RPC::emit(DeviceLoop* loop, Code code)
{
    syncWrite(&loop->m_codes[position], static_cast<int32_t>(code));
}

template <typename DeviceLoop>
GLOOP_ALWAYS_INLINE __device__ Code RPC::peek(DeviceLoop* loop)
{
    return readNoCache<Code>(&loop->m_codes[position]);
}

template <typename DeviceLoop>
GLOOP_ALWAYS_INLINE __device__ request::Payload* RPC::request(DeviceLoop* loop)
{
    return &loop->m_payloads[position];
}

template <typename Policy>
inline __device__ uint32_t DeviceLoop<Policy>::position(OnePage* page)
{
    GLOOP_ASSERT_SINGLE_THREAD();
    return page - pages();
}

template <typename Policy>
inline __device__ bool DeviceLoop<Policy>::isValidPosition(uint32_t position)
{
    return position < GLOOP_SHARED_SLOT_SIZE;
}

template <>
GLOOP_ALWAYS_INLINE __device__ auto DeviceLoop<Shared>::slots(uint32_t position) -> DeviceCallback*
{
    GLOOP_ASSERT_SINGLE_THREAD();
#if defined(GLOOP_ENABLE_HIERARCHICAL_SLOT_MEMORY)
    if (position == m_special.m_scratchIndex1) {
        return reinterpret_cast<DeviceCallback*>(&m_special.m_scratch1);
    }
    if (position == m_special.m_scratchIndex2) {
        return reinterpret_cast<DeviceCallback*>(&m_special.m_scratch2);
    }
#endif
    return m_slots + position;
}

template <>
GLOOP_ALWAYS_INLINE __device__ auto DeviceLoop<Global>::slots(uint32_t position) -> DeviceCallback*
{
    GLOOP_ASSERT_SINGLE_THREAD();
    return m_slots + position;
}

template <typename Policy>
inline __device__ auto DeviceLoop<Policy>::context() -> PerBlockContext*
{
    GLOOP_ASSERT_SINGLE_THREAD();
    return m_deviceContext->context + GLOOP_BID();
}

template <typename Policy>
inline __device__ auto DeviceLoop<Policy>::hostContext() -> PerBlockHostContext*
{
    GLOOP_ASSERT_SINGLE_THREAD();
    return m_deviceContext->hostContext + GLOOP_BID();
}

template <typename Policy>
inline __device__ auto DeviceLoop<Policy>::pages() const -> OnePage*
{
    GLOOP_ASSERT_SINGLE_THREAD();
    return m_pages;
}

template <typename Policy>
template <typename Lambda>
inline __device__ void DeviceLoop<Policy>::allocOnePage(Lambda&& lambda)
{
    __shared__ void* page;
    BEGIN_SINGLE_THREAD
    {
        page = nullptr;
        int freePagePosPlusOne = __ffs(m_control.freePages);
        if (freePagePosPlusOne == 0) {
            uint32_t pos = enqueueSleep([lambda](DeviceLoop<Policy>* loop, volatile request::Request* req) {
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

template <typename Policy>
template <typename Lambda>
inline __device__ void DeviceLoop<Policy>::enqueueLater(Lambda&& lambda)
{
    GLOOP_ASSERT_SINGLE_THREAD();
    uint32_t pos = enqueueSleep(std::forward<Lambda&&>(lambda));
    m_control.wakeupSlots |= (1U << pos);
}

template <typename Policy>
template <typename Lambda>
inline __device__ uint32_t DeviceLoop<Policy>::enqueueSleep(Lambda&& lambda)
{
    GLOOP_ASSERT_SINGLE_THREAD();
    uint32_t pos = allocate(std::forward<Lambda&&>(lambda));
    m_control.sleepSlots |= (1U << pos);
    m_control.wakeupSlots &= ~(1U << pos);
    return pos;
}

template <typename Policy>
template <typename Lambda>
inline __device__ RPC DeviceLoop<Policy>::enqueueRPC(Lambda&& lambda)
{
    GLOOP_ASSERT_SINGLE_THREAD();
    uint32_t pos = allocate(std::forward<Lambda&&>(lambda));
    GPU_ASSERT(pos < GLOOP_SHARED_SLOT_SIZE);
    return {pos};
}

template <>
inline __device__ void* DeviceLoop<Global>::allocateSharedSlotIfNecessary(uint32_t pos)
{
    GLOOP_ASSERT_SINGLE_THREAD();
    return m_slots + pos;
}

template <>
inline __device__ void* DeviceLoop<Shared>::allocateSharedSlotIfNecessary(uint32_t pos)
{
    GLOOP_ASSERT_SINGLE_THREAD();
#if defined(GLOOP_ENABLE_HIERARCHICAL_SLOT_MEMORY)
    if (m_special.m_scratchIndex1 == invalidPosition()) {
        m_special.m_scratchIndex1 = pos;
        return &m_special.m_scratch1;
    }
    if (m_special.m_scratchIndex2 == invalidPosition()) {
        m_special.m_scratchIndex2 = pos;
        return &m_special.m_scratch2;
    }
#endif
    return m_slots + pos;
}

template <typename Policy>
template <typename Lambda>
inline __device__ uint32_t DeviceLoop<Policy>::allocate(Lambda&& lambda)
{
    GLOOP_ASSERT_SINGLE_THREAD();
    int pos = __ffs(m_control.freeSlots) - 1;
    GPU_ASSERT(pos >= 0 && pos <= GLOOP_SHARED_SLOT_SIZE);
    GPU_ASSERT(m_control.freeSlots & (1U << pos));
    m_control.freeSlots &= ~(1U << pos);

    void* target = allocateSharedSlotIfNecessary(pos);

    new (target) DeviceCallback(std::forward<Lambda&&>(lambda));

    return pos;
}

template <typename Policy>
inline __device__ auto DeviceLoop<Policy>::dequeue() -> uint32_t
{
    GLOOP_ASSERT_SINGLE_THREAD();

    uint32_t freeSlots = m_control.freeSlots;
    if (freeSlots == DeviceLoopControl::allFilledFreeSlots()) {
        return shouldExitPosition();
    }

    // __threadfence_system();
    // We first search wake up slots. It is always ready to execute.
    // And we can get the slot without costly DMA.
    uint32_t allocatedSlots = freeSlots ^ DeviceLoopControl::allFilledFreeSlots();
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
            RPC rpc{i};
            Code code = rpc.peek(this);
            if (code == Code::Complete) {
                rpc.emit(this, Code::None);
                GPU_ASSERT(rpc.peek(this) != Code::Complete);
                GPU_ASSERT(rpc.peek(this) == Code::None);
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

template <>
inline __device__ void DeviceLoop<Global>::deallocateSharedSlotIfNecessary(uint32_t pos)
{
    GLOOP_ASSERT_SINGLE_THREAD();
}

template <>
inline __device__ void DeviceLoop<Shared>::deallocateSharedSlotIfNecessary(uint32_t pos)
{
    // We are using one shot function. After calling the function, destruction is already done.
    // callback->~DeviceCallback();
    GLOOP_ASSERT_SINGLE_THREAD();
#if defined(GLOOP_ENABLE_HIERARCHICAL_SLOT_MEMORY)
    if (pos == m_special.m_scratchIndex1) {
        m_special.m_scratchIndex1 = invalidPosition();
    } else if (pos == m_special.m_scratchIndex2) {
        m_special.m_scratchIndex2 = invalidPosition();
    }
#endif
}

template <typename Policy>
inline __device__ void DeviceLoop<Policy>::deallocate(uint32_t pos)
{
    GLOOP_ASSERT_SINGLE_THREAD();
    // printf("pos:(%u)\n", (unsigned)pos);
    GPU_ASSERT(pos >= 0 && pos <= GLOOP_SHARED_SLOT_SIZE);
    GPU_ASSERT(!(m_control.freeSlots & (1U << pos)));
    deallocateSharedSlotIfNecessary(pos);
    m_control.freeSlots |= (1U << pos);
}

template <typename Policy>
inline __device__ int DeviceLoop<Policy>::shouldPostTask()
{
    GLOOP_ASSERT_SINGLE_THREAD();
    return (clock64() - m_start) > m_killClock;
}

template <>
inline __device__ int DeviceLoop<Global>::drain()
{
    uint32_t position = shouldExitPosition();

    BEGIN_SINGLE_THREAD
    {
        m_special.m_nextCallback = nullptr;
        if (m_control.freeSlots != DeviceLoopControl::allFilledFreeSlots()) {
            position = invalidPosition();
        }
    }
    END_SINGLE_THREAD

    while (__syncthreads_or(position != shouldExitPosition())) {
        if (m_special.m_nextCallback) {
            // One shot function always destroys the function and syncs threads.
            (*reinterpret_cast<DeviceCallback*>(m_special.m_nextCallback))(this, m_special.m_nextPayload);
        }

        __syncthreads();
        BEGIN_SINGLE_THREAD_WITHOUT_BARRIER
        {
            if (position != invalidPosition()) {
                deallocate(position);
            }

            {
                uint64_t now = clock64();
                if (((now - m_start) > m_killClock)) {
                    m_start = ((now / m_killClock) * m_killClock);
                    if (gloop::readNoCache<uint32_t>(signal) != 0) {
                        position = shouldExitPosition();
                        goto next;
                    }
                }
            }

            position = dequeue();
            if (isValidPosition(position)) {
                m_special.m_nextCallback = slots(position);
                m_special.m_nextPayload = &m_payloads[position];
            } else {
                m_special.m_nextCallback = nullptr;
            }
        next:
        }
        END_SINGLE_THREAD_WITHOUT_BARRIER
    }

    // CAUTION: Do not use shared memory to broadcast the result.
    // We use __syncthreads_or carefully here to scatter the boolean value.
    int suspended = 0;
    __syncthreads();
    BEGIN_SINGLE_THREAD_WITHOUT_BARRIER
    {
        suspended = suspend();
    }
    END_SINGLE_THREAD_WITHOUT_BARRIER
    return __syncthreads_or(suspended);
}

template <>
inline __device__ int DeviceLoop<Shared>::drain()
{
    __shared__ uint32_t position;
    __shared__ DeviceCallback* callback;

    BEGIN_SINGLE_THREAD
    {
        callback = nullptr;
        if (m_control.freeSlots == DeviceLoopControl::allFilledFreeSlots()) {
            position = shouldExitPosition();
        } else {
            position = invalidPosition();
        }
    }
    END_SINGLE_THREAD

    while (position != shouldExitPosition()) {
        if (callback) {
            // One shot function always destroys the function and syncs threads.
            (*callback)(this, &m_payloads[position]);
        }

        BEGIN_SINGLE_THREAD
        {
            if (callback) {
                deallocate(position);
            }

            {
                uint64_t now = clock64();
                if (((now - m_start) > m_killClock)) {
                    m_start = ((now / m_killClock) * m_killClock);
                    if (gloop::readNoCache<uint32_t>(signal) != 0) {
                        position = shouldExitPosition();
                        goto next;
                    }
                }
            }

            callback = nullptr;
            position = dequeue();
            if (isValidPosition(position)) {
                callback = slots(position);
            }
        next:
        }
        END_SINGLE_THREAD
    }

    // CAUTION: Do not use shared memory to broadcast the result.
    // We use __syncthreads_or carefully here to scatter the boolean value.
    int suspended = 0;
    __syncthreads();
    BEGIN_SINGLE_THREAD_WITHOUT_BARRIER
    {
        suspended = suspend();
    }
    END_SINGLE_THREAD_WITHOUT_BARRIER
    return suspended;
}

template <>
inline __device__ void DeviceLoop<Global>::suspendSharedSlots()
{
}

template <>
inline __device__ void DeviceLoop<Shared>::suspendSharedSlots()
{
#if defined(GLOOP_ENABLE_HIERARCHICAL_SLOT_MEMORY)
    if (m_special.m_scratchIndex1 != invalidPosition()) {
        new (m_slots + m_special.m_scratchIndex1) DeviceCallback(*reinterpret_cast<DeviceCallback*>(&m_special.m_scratch1));
        reinterpret_cast<DeviceCallback*>(&m_special.m_scratch1)->~DeviceCallback();
    }
    if (m_special.m_scratchIndex2 != invalidPosition()) {
        new (m_slots + m_special.m_scratchIndex2) DeviceCallback(*reinterpret_cast<DeviceCallback*>(&m_special.m_scratch2));
        reinterpret_cast<DeviceCallback*>(&m_special.m_scratch2)->~DeviceCallback();
    }
#endif
}

template <typename Policy>
inline __device__ int DeviceLoop<Policy>::suspend()
{
    GLOOP_ASSERT_SINGLE_THREAD();
    if (m_control.freeSlots != DeviceLoopControl::allFilledFreeSlots()) {
        // Save the control state.
        PerBlockContext* blockContext = context();
        blockContext->control = m_control;
        blockContext->logicalBlockIdx = logicalBlockIdxInternal();
        blockContext->logicalGridDim = logicalGridDimInternal();
        PerBlockHostContext* hContext = hostContext();
        hContext->freeSlots = m_control.freeSlots;
        hContext->sleepSlots = m_control.sleepSlots;
        hContext->wakeupSlots = m_control.wakeupSlots;

        suspendSharedSlots();

        // Request the resume.
        atomicAdd(&m_kernel->pending, 1);
        return /* stop the loop */ 1;
    }

#if defined(GLOOP_ENABLE_ELASTIC_KERNELS)
    // This logical thread block is done.
    if (--m_control.logicalBlocksCount != 0) {
        // There is some remaining logical thread blocks.
        // Let's increment the logical block index.
        logicalBlockIdxInternal().x += 1;
        if (logicalBlockIdxInternal().x == logicalGridDimInternal().x) {
            logicalBlockIdxInternal().x = 0;
            logicalBlockIdxInternal().y += 1;
        }

        uint64_t now = clock64();
        if (((now - m_start) > m_killClock)) {
            m_start = ((now / m_killClock) * m_killClock);
            if (gloop::readNoCache<uint32_t>(signal) != 0) {

                // Save the control state.
                PerBlockContext* blockContext = context();
                blockContext->control = m_control;
                blockContext->logicalBlockIdx = logicalBlockIdxInternal();
                blockContext->logicalGridDim = logicalGridDimInternal();
                PerBlockHostContext* hContext = hostContext();
                hContext->freeSlots = DeviceLoopControl::allFilledFreeSlots();
                hContext->sleepSlots = 0;
                hContext->wakeupSlots = 0;

                // Request the resume.
                atomicAdd(&m_kernel->pending, 1);
                return /* stop the loop */ 1;
            }
        }

        return /* continue the next loop */ 0;
    }
#endif

    // Save the control state. We need to save the control state since
    // the other thread block may not stop yet. In that case, this
    // this block may be re-launched.
    PerBlockContext* blockContext = context();
    blockContext->control = m_control;
    blockContext->logicalBlockIdx = logicalBlockIdxInternal();
    blockContext->logicalGridDim = logicalGridDimInternal();
    PerBlockHostContext* hContext = hostContext();
    hContext->freeSlots = DeviceLoopControl::allFilledFreeSlots();
    hContext->sleepSlots = 0;
    hContext->wakeupSlots = 0;

    // Finish the whole kernel.
    return /* stop the loop */ 1;
}

template <>
inline __device__ void DeviceLoop<Global>::initializeSharedSlots()
{
}

template <>
inline __device__ void DeviceLoop<Shared>::initializeSharedSlots()
{
#if defined(GLOOP_ENABLE_HIERARCHICAL_SLOT_MEMORY)
    m_special.m_scratchIndex1 = invalidPosition();
    m_special.m_scratchIndex2 = invalidPosition();
#endif
}

template <typename Policy>
inline __device__ void DeviceLoop<Policy>::initializeImpl(const DeviceContext& deviceContext)
{
    GLOOP_ASSERT_SINGLE_THREAD();

    m_deviceContext = &deviceContext;
    m_codes = deviceContext.codes + (GLOOP_BID() * GLOOP_SHARED_SLOT_SIZE);
    m_payloads = deviceContext.payloads + (GLOOP_BID() * GLOOP_SHARED_SLOT_SIZE);
    m_pages = deviceContext.pages + (GLOOP_BID() * GLOOP_SHARED_PAGE_COUNT);
    m_kernel = deviceContext.kernel;
    m_killClock = deviceContext.killClock;

    uint64_t startClock = clock64();
    m_start = atomicCAS((unsigned long long*)&m_kernel->globalClock, 0ULL, (unsigned long long)startClock);
    if (m_start == 0)
        m_start = startClock;

    m_slots = reinterpret_cast<DeviceCallback*>(&context()->slots);
    initializeSharedSlots();
}

template <typename Policy>
inline __device__ void DeviceLoop<Policy>::initialize(const DeviceContext& deviceContext)
{
    GLOOP_ASSERT_SINGLE_THREAD();
    initializeImpl(deviceContext);
    uint3 logicalBlocksDim = deviceContext.logicalBlocks;
    m_control.initialize(logicalBlocksDim);
#if defined(GLOOP_ENABLE_ELASTIC_KERNELS)
    logicalBlockIdxInternal() = make_uint2(m_control.currentLogicalBlockCount % logicalBlocksDim.x, m_control.currentLogicalBlockCount / logicalBlocksDim.x);
    logicalGridDimInternal() = make_uint2(logicalBlocksDim.x, logicalBlocksDim.y);
#endif
}

template <typename Policy>
inline __device__ int DeviceLoop<Policy>::initialize(const DeviceContext& deviceContext, ResumeTag)
{
    GLOOP_ASSERT_SINGLE_THREAD();
    initializeImpl(deviceContext);
    resume();
#if defined(GLOOP_ENABLE_ELASTIC_KERNELS)
    return m_control.freeSlots != DeviceLoopControl::allFilledFreeSlots();
#else
    return 1;
#endif
}

template <typename Policy>
inline __device__ void DeviceLoop<Policy>::resume()
{
    GLOOP_ASSERT_SINGLE_THREAD();
    // __threadfence_system();  // FIXME
    PerBlockContext* blockContext = context();
    m_control = blockContext->control;

#if defined(GLOOP_ENABLE_ELASTIC_KERNELS)
    logicalBlockIdxInternal() = blockContext->logicalBlockIdx;
    logicalGridDimInternal() = blockContext->logicalGridDim;
#endif

    // __threadfence_system();  // FIXME
}

template <typename Policy>
inline __device__ void DeviceLoop<Policy>::freeOnePage(void* aPage)
{
    GLOOP_ASSERT_SINGLE_THREAD();
    uint32_t pos = position(static_cast<OnePage*>(aPage));
    m_control.freePages |= (1UL << pos);
    GPU_ASSERT(pos < GLOOP_SHARED_PAGE_COUNT);
    int freePageWaitingCallbackPlusOne = __ffs(m_control.pageSleepSlots);
    if (freePageWaitingCallbackPlusOne) {
        m_control.wakeupSlots |= (1U << (freePageWaitingCallbackPlusOne - 1));
    }
}

template <typename Policy>
GLOOP_ALWAYS_INLINE __device__ const uint2& DeviceLoop<Policy>::logicalBlockIdx() const
{
    return m_logicalBlockIdx;
}

template <typename Policy>
GLOOP_ALWAYS_INLINE __device__ const uint2& DeviceLoop<Policy>::logicalGridDim() const
{
    return m_logicalGridDim;
}

template <typename Policy>
GLOOP_ALWAYS_INLINE __device__ uint2& DeviceLoop<Policy>::logicalBlockIdxInternal()
{
    return m_logicalBlockIdx;
}

template <typename Policy>
GLOOP_ALWAYS_INLINE __device__ uint2& DeviceLoop<Policy>::logicalGridDimInternal()
{
    return m_logicalGridDim;
}

} // namespace gloop
