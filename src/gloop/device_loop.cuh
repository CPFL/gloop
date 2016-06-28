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
#ifndef GLOOP_DEVICE_LOOP_H_
#define GLOOP_DEVICE_LOOP_H_
#include <cstdint>
#include <type_traits>
#include "code.cuh"
#include "config.h"
#include "device_callback.cuh"
#include "device_context.cuh"
#include "one_shot_function.cuh"
#include "rpc.cuh"
#include "request.h"
#include "utility.h"
#include "utility.cuh"
#include "utility/util.cu.h"
namespace gloop {

struct RPC;

class DeviceLoop {
public:
    friend struct RPC;
    enum ResumeTag { Resume };
    inline __device__ int initialize(DeviceContext, ResumeTag);
    inline __device__ void initialize(DeviceContext);

    template<typename Lambda>
    inline __device__ RPC enqueueRPC(Lambda&& lambda);
    template<typename Lambda>
    inline __device__ void enqueueLater(Lambda&& lambda);

    template<typename Lambda>
    inline __device__ void allocOnePage(Lambda&& lambda);
    inline __device__ void freeOnePage(void* page);

    inline __device__ int drain();

#if defined(GLOOP_ENABLE_ELASTIC_KERNELS)
    // GLOOP_ALWAYS_INLINE __device__ auto logicalBlockIdx() -> uint2 const { return m_control.logicalBlockIdx; }
    // GLOOP_ALWAYS_INLINE __device__ auto logicalBlockIdxX() -> unsigned const { return m_control.logicalBlockIdx.x; }
    // GLOOP_ALWAYS_INLINE __device__ auto logicalBlockIdxY() -> unsigned const { return m_control.logicalBlockIdx.y; }

    // GLOOP_ALWAYS_INLINE __device__ auto logicalGridDim() -> uint2 const { return m_control.logicalGridDim; }
    // GLOOP_ALWAYS_INLINE __device__ auto logicalGridDimX() -> unsigned const { return m_control.logicalGridDim.x; }
    // GLOOP_ALWAYS_INLINE __device__ auto logicalGridDimY() -> unsigned const { return m_control.logicalGridDim.y; }

    GLOOP_ALWAYS_INLINE __device__ unsigned logicalBlocksCount() const { return m_control.logicalBlocksCount; }
#endif

    GLOOP_ALWAYS_INLINE __device__ int shouldPostTask();

private:
    inline __device__ void initializeImpl(DeviceContext);

    template<typename Lambda>
    inline __device__ uint32_t enqueueSleep(Lambda&& lambda);

    template<typename Lambda>
    inline __device__ uint32_t allocate(Lambda&& lambda);

    inline __device__ void deallocate(uint32_t pos);

    inline __device__ uint32_t dequeue();

    inline __device__ void resume();
    inline __device__ int suspend();

    GLOOP_ALWAYS_INLINE __device__ DeviceCallback* slots(uint32_t position);
    GLOOP_ALWAYS_INLINE __device__ PerBlockContext* context() const;
    GLOOP_ALWAYS_INLINE __device__ OnePage* pages() const;
    GLOOP_ALWAYS_INLINE __device__ uint32_t position(OnePage*);

    __device__ static constexpr uint32_t shouldExitPosition() { return UINT32_MAX - 1; }
    __device__ static constexpr uint32_t invalidPosition() { return UINT32_MAX; }
    GLOOP_ALWAYS_INLINE __device__ static bool isValidPosition(uint32_t position);

    PerBlockContext* m_context;
    PerBlockHostContext* m_hostContext;

    // SoA.
    int32_t* m_codes;
    request::Payload* m_payloads;
    OnePage* m_pages;

    // FIXME: Global case only.
    DeviceCallback* m_nextCallback;
    request::Payload* m_nextPayload;

    KernelContext* m_kernel;
    uint64_t m_killClock;
    uint64_t m_start;

    DeviceCallback* m_slots;
    DeviceLoopControl m_control;

#if defined(GLOOP_ENABLE_HIERARCHICAL_SLOT_MEMORY)
    uint32_t m_scratchIndex1;
    uint32_t m_scratchIndex2;
    UninitializedDeviceCallbackStorage m_scratch1;
    UninitializedDeviceCallbackStorage m_scratch2;
#endif
};
static_assert(std::is_trivially_destructible<DeviceLoop>::value, "DeviceLoop is trivially destructible");

extern __device__ __shared__ uint2 logicalGridDim;
extern __device__ __shared__ uint2 logicalBlockIdx;
extern __device__ volatile uint32_t* signal;

}  // namespace gloop
#endif  // GLOOP_DEVICE_LOOP_H_
