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
#include "ipc.cuh"
#include "request.h"
#include "utility.h"
#include "utility.cuh"
#include "utility/util.cu.h"
namespace gloop {

struct IPC;

class DeviceLoop {
public:
    friend struct IPC;
    enum ResumeTag { Resume };
    __device__ int initialize(volatile uint32_t* signal, DeviceContext, ResumeTag);
    __device__ void initialize(volatile uint32_t* signal, DeviceContext);

    template<typename Lambda>
    inline __device__ IPC enqueueIPC(Lambda&& lambda);
    template<typename Lambda>
    inline __device__ void enqueueLater(Lambda&& lambda);

    template<typename Lambda>
    inline __device__ void allocOnePage(Lambda&& lambda);
    __device__ void freeOnePage(void* page);

    __device__ int drain(int executeAtLeastOne);

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
    __device__ void initializeImpl(DeviceContext);

    template<typename Lambda>
    inline __device__ uint32_t enqueueSleep(Lambda&& lambda);

    template<typename Lambda>
    inline __device__ uint32_t allocate(Lambda&& lambda);

    inline __device__ void deallocate(uint32_t pos);

    inline __device__ uint32_t dequeue();

    __device__ void resume();
    __device__ int suspend();

    GLOOP_ALWAYS_INLINE __device__ DeviceCallback* slots(uint32_t position);
    GLOOP_ALWAYS_INLINE __device__ DeviceContext::PerBlockContext* context() const;
    GLOOP_ALWAYS_INLINE __device__ DeviceContext::OnePage* pages() const;
    GLOOP_ALWAYS_INLINE __device__ uint32_t position(DeviceContext::OnePage*);

    __device__ static constexpr uint32_t shouldExitPosition() { return UINT32_MAX - 1; }
    __device__ static constexpr uint32_t invalidPosition() { return UINT32_MAX; }
    GLOOP_ALWAYS_INLINE __device__ static bool isValidPosition(uint32_t position);

    DeviceContext m_deviceContext;

    // SoA.
    int32_t* m_codes;
    request::Payload* m_payloads;

    DeviceCallback* m_slots;
    DeviceContext::DeviceLoopControl m_control;
    uint64_t m_start;

#if defined(GLOOP_ENABLE_HIERARCHICAL_SLOT_MEMORY)
    uint32_t m_scratchIndex1;
    uint32_t m_scratchIndex2;
    UninitializedDeviceCallbackStorage m_scratch1;
    UninitializedDeviceCallbackStorage m_scratch2;
#endif
};
static_assert(std::is_trivially_destructible<DeviceLoop>::value, "DeviceLoop is trivially destructible");

extern __device__ __shared__ DeviceLoop sharedDeviceLoop;
extern __device__ __shared__ uint2 logicalGridDim;
extern __device__ __shared__ uint2 logicalBlockIdx;
extern __device__ UninitializedDeviceCallbackStorage nextKernel;

}  // namespace gloop
#endif  // GLOOP_DEVICE_LOOP_H_
