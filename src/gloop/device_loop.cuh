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
  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
  THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include "code.cuh"
#include "config.h"
#include "device_callback.cuh"
#include "device_data.cuh"
#include "request.h"
#include "rpc.cuh"
#include "utility.cuh"
#include "utility.h"
#include "utility/util.cu.h"
#include <cstdint>
#include <type_traits>
namespace gloop {

struct DeviceContext;

template <typename Policy>
struct DeviceLoopSpecialData;

template <>
struct DeviceLoopSpecialData<Shared> {
#if defined(GLOOP_ENABLE_HIERARCHICAL_SLOT_MEMORY)
    uint32_t m_scratchIndex1;
    uint32_t m_scratchIndex2;
    UninitializedDeviceCallbackStorage m_scratch1;
    UninitializedDeviceCallbackStorage m_scratch2;
#endif
};

template <>
struct DeviceLoopSpecialData<Global> {
    void* m_nextCallback;
    request::Payload* m_nextPayload;
};

template <typename Policy = Shared>
class DeviceLoop {
public:
    typedef gloop::OneShotFunction<void(DeviceLoop<Policy>*, volatile request::Request*)> DeviceCallback;
    static_assert(sizeof(DeviceCallback) == sizeof(UninitializedDeviceCallbackStorage), "DeviceCallback size check.");

    friend struct RPC;
    enum ResumeTag { Resume };
    inline __device__ int initialize(const DeviceContext&, ResumeTag);
    inline __device__ void initialize(const DeviceContext&);

    template <typename Lambda>
    inline __device__ RPC enqueueRPC(Lambda&& lambda);
    template <typename Lambda>
    inline __device__ void enqueueLater(Lambda&& lambda);

    template <typename Lambda>
    inline __device__ void allocOnePage(Lambda&& lambda);
    inline __device__ void freeOnePage(void* page);

    template<typename ThreadBlock>
    inline __device__ int drain(ThreadBlock);

    GLOOP_ALWAYS_INLINE __device__ const uint2& logicalBlockIdx() const;

    GLOOP_ALWAYS_INLINE __device__ const uint2& logicalGridDim() const;

    GLOOP_ALWAYS_INLINE __device__ unsigned logicalBlocksCount() const
    {
        return m_control.logicalBlocksCount;
    }

    GLOOP_ALWAYS_INLINE __device__ int isLastLogicalBlock() const
    {
        return m_control.logicalBlocksCount == 1;
    }

    GLOOP_ALWAYS_INLINE __device__ int shouldPostTask();

    GLOOP_ALWAYS_INLINE __device__ void*& scratch()
    {
        return m_control.scratch;
    }

private:
    GLOOP_ALWAYS_INLINE __device__ uint2& logicalBlockIdxInternal();

    GLOOP_ALWAYS_INLINE __device__ uint2& logicalGridDimInternal();

    inline __device__ void initializeImpl(const DeviceContext&);

    template <typename Lambda>
    inline __device__ uint32_t enqueueSleep(Lambda&& lambda);

    template <typename Lambda>
    inline __device__ uint32_t allocate(Lambda&& lambda);

    inline __device__ void deallocate(uint32_t pos);

    inline __device__ uint32_t dequeue();

    inline __device__ void resume();
    inline __device__ int suspend();

    inline __device__ void* allocateSharedSlotIfNecessary(uint32_t position);
    inline __device__ void deallocateSharedSlotIfNecessary(uint32_t pos);
    inline __device__ void initializeSharedSlots();
    inline __device__ void suspendSharedSlots();

    GLOOP_ALWAYS_INLINE __device__ DeviceCallback* slots(uint32_t position);
    GLOOP_ALWAYS_INLINE __device__ PerBlockContext* context();
    GLOOP_ALWAYS_INLINE __device__ PerBlockHostContext* hostContext();
    GLOOP_ALWAYS_INLINE __device__ OnePage* pages() const;
    GLOOP_ALWAYS_INLINE __device__ uint32_t position(OnePage*);

    __device__ static constexpr uint32_t newExecutionPosition()
    {
        return UINT32_MAX - 2;
    }
    __device__ static constexpr uint32_t shouldExitPosition()
    {
        return UINT32_MAX - 1;
    }
    __device__ static constexpr uint32_t invalidPosition()
    {
        return UINT32_MAX;
    }

    GLOOP_ALWAYS_INLINE __device__ static bool isValidPosition(uint32_t position);

    bool dequeueThreadBlock(DeviceThreadBlock&);

    const DeviceContext* m_deviceContext;

    // SoA.
    int32_t* m_codes;
    request::Payload* m_payloads;
    OnePage* m_pages;

    KernelContext* m_kernel;
    uint64_t m_killClock;
    uint64_t m_start;

    DeviceCallback* m_slots;
    DeviceLoopControl m_control;

    DeviceThreadBlock m_block1;
    DeviceThreadBlock m_block2;
    DeviceThreadBlock* m_currentBlock;
    uint2 m_logicalGridDim;

    DeviceLoopSpecialData<Policy> m_special;
};

extern __device__ volatile uint32_t* signal;

} // namespace gloop
