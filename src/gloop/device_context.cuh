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
#ifndef GLOOP_DEVICE_CONTEXT_CU_H_
#define GLOOP_DEVICE_CONTEXT_CU_H_
#include <cstdint>
#include <type_traits>
#include "config.h"
#include "device_callback.cuh"
namespace gloop {

class IPC;

struct DeviceContext {
    static_assert(GLOOP_SHARED_PAGE_COUNT < 32, "Should be less than 32");
    struct DeviceLoopControl {
        __host__ __device__ static constexpr uint32_t allFilledFreeSlots()
        {
            return ((1U << GLOOP_SHARED_SLOT_SIZE) - 1);
        }

        __host__ __device__ void initialize(uint3 logicalBlocksDim)
        {
            freePages = (1U << GLOOP_SHARED_PAGE_COUNT) - 1;
            freeSlots = allFilledFreeSlots();
            sleepSlots = 0;
            wakeupSlots = 0;
            pageSleepSlots = 0;

            // Calculate the logical blocks per physical blocks.
            uint32_t logicalBlocks = logicalBlocksDim.x * logicalBlocksDim.y;
            uint32_t physicalBlocks = GLOOP_BMAX();
            uint32_t count = logicalBlocks / physicalBlocks;
            uint32_t remaining = logicalBlocks % physicalBlocks;
            uint32_t bid = GLOOP_BID();
            if (remaining > bid) {
                count += 1;
                currentLogicalBlockCount = count * bid;
            } else {
                currentLogicalBlockCount = count * bid + remaining;
            }
            logicalBlocksCount = count;

            logicalBlockIdx = make_uint2(currentLogicalBlockCount % logicalBlocksDim.x, currentLogicalBlockCount / logicalBlocksDim.x);
            logicalGridDim = make_uint2(logicalBlocksDim.x, logicalGridDim.y);
        }

        uint32_t freePages;
        uint32_t freeSlots;
        uint32_t sleepSlots;
        uint32_t wakeupSlots;
        uint32_t pageSleepSlots;
        uint32_t logicalBlocksCount;
        uint32_t currentLogicalBlockCount;
        uint2 logicalBlockIdx;
        uint2 logicalGridDim;
    };

    struct OnePage {
        unsigned char data[GLOOP_SHARED_PAGE_SIZE];
    };

    struct PerBlockContext {
        static const std::size_t PerBlockSize = GLOOP_SHARED_SLOT_SIZE * sizeof(UninitializedDeviceCallbackStorage);
        typedef std::aligned_storage<PerBlockSize>::type Slots;
        Slots slots;
        DeviceLoopControl control;
    };

    struct KernelContext {
        uint32_t pending;
        uint64_t globalClock;
    };

    PerBlockContext* context;
    IPC* channels;
    OnePage* pages;
    KernelContext* kernel;
    uint64_t killClock;
    uint3 logicalBlocks;
};

}  // namespace gloop
#endif  // GLOOP_DEVICE_CONTEXT_CU_H_
