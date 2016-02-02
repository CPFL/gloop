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
#ifndef GLOOP_DEVICE_LOOP_H_
#define GLOOP_DEVICE_LOOP_H_
#include <cstdint>
#include <gipc/gipc.cuh>
#include <gpufs/libgpufs/util.cu.h>
#include "code.cuh"
#include "function.cuh"
#include "ipc.cuh"
#include "request.h"
#include "utility.h"
namespace gloop {

#define GLOOP_SHARED_SLOT_SIZE 64
#define GLOOP_SHARED_PAGE_SIZE 4096UL
#define GLOOP_SHARED_PAGE_COUNT 2

__device__ extern IPC* g_channel;

class DeviceLoop {
public:
    typedef gloop::function<void(DeviceLoop*, volatile request::Request*)> Callback;
    typedef std::aligned_storage<sizeof(DeviceLoop::Callback), alignof(DeviceLoop::Callback)>::type UninitializedStorage;

    struct OnePage {
        unsigned char data[GLOOP_SHARED_PAGE_SIZE];
    };

    static_assert(GLOOP_SHARED_PAGE_COUNT != 64, "Should not be 64");
    struct DeviceLoopControl {
        uint32_t put { 0 };
        uint32_t get { 0 };
        uint32_t pending { 0 };
        uint64_t free { static_cast<decltype(free)>(-1) };
        uint64_t sleep { 0 };
        uint64_t wakeup { 0 };
        uint64_t freePages { (1ULL << GLOOP_SHARED_PAGE_COUNT) - 1 };
        uint64_t m_pageSleep { 0 };
        uint8_t queue[GLOOP_SHARED_SLOT_SIZE];
    };

    struct PerBlockContext {
        static const std::size_t PerBlockSize = GLOOP_SHARED_SLOT_SIZE * sizeof(UninitializedStorage);
        typedef std::aligned_storage<PerBlockSize>::type Slots;
        Slots slots;
        DeviceLoopControl control;
    };

    struct DeviceContext {
        PerBlockContext* context;
        IPC* channels;
        OnePage* pages;
        uint32_t* pending;
    };

    __device__ DeviceLoop(volatile uint32_t* signal, DeviceContext, size_t size);

    __device__ IPC* enqueueIPC(Callback lambda);
    __device__ void enqueueLater(Callback lambda);

    template<typename Lambda>
    __device__ void allocOnePageMaySync(Lambda lambda);
    __device__ void freeOnePage(void* page);

    __device__ void emit(Code code, IPC*);

    __device__ Callback* dequeue();

    __device__ bool drain();

    __device__ void deallocate(Callback* callback);

    __device__ void resume();

private:
    __device__ uint32_t enqueueSleep(const Callback& lambda);

    __device__ uint32_t allocate(const Callback* lambda);

    __device__ void suspend();

    GLOOP_ALWAYS_INLINE __device__ IPC* channel() const;
    GLOOP_ALWAYS_INLINE __device__ PerBlockContext* context() const;
    GLOOP_ALWAYS_INLINE __device__ OnePage* pages() const;
    GLOOP_ALWAYS_INLINE __device__ uint32_t position(Callback*);
    GLOOP_ALWAYS_INLINE __device__ uint32_t position(IPC*);
    GLOOP_ALWAYS_INLINE __device__ uint32_t position(OnePage*);

    DeviceContext m_deviceContext;
    Callback* m_slots;
    DeviceLoopControl m_control;
    volatile uint32_t* m_signal;
};

__device__ uint32_t DeviceLoop::position(Callback* callback)
{
    return callback - m_slots;
}

__device__ uint32_t DeviceLoop::position(IPC* ipc)
{
    return ipc - channel();
}

GLOOP_ALWAYS_INLINE __device__ uint32_t DeviceLoop::position(OnePage* page)
{
    return page - pages();
}

__device__ auto DeviceLoop::channel() const -> IPC*
{
    return m_deviceContext.channels + (GLOOP_BID() * GLOOP_SHARED_SLOT_SIZE);
}

__device__ auto DeviceLoop::context() const -> PerBlockContext*
{
    return m_deviceContext.context + GLOOP_BID();
}

__device__ auto DeviceLoop::pages() const -> OnePage*
{
    return m_deviceContext.pages + (GLOOP_BID() * GLOOP_SHARED_PAGE_COUNT);
}

template<typename Lambda>
inline __device__ void DeviceLoop::allocOnePageMaySync(Lambda lambda)
{
    __shared__ void* page;
    BEGIN_SINGLE_THREAD
    {
        page = nullptr;
        int freePagePosPlusOne = __ffsll(m_control.freePages);
        if (freePagePosPlusOne == 0) {
            uint32_t pos = enqueueSleep([lambda](DeviceLoop* loop, volatile request::Request* req) {
                loop->allocOnePageMaySync(lambda);
            });
            m_control.m_pageSleep |= (1ULL << pos);
        } else {
            int pagePos = freePagePosPlusOne - 1;
            page = pages() + pagePos;
            m_control.freePages &= ~(1ULL << pagePos);
#if 0
            enqueueLater([lambda, page](DeviceLoop* loop, volatile request::Request* req) {
                volatile request::Request request;
                request.u.allocOnePageResult.page = page;
                lambda(loop, &request);
            });
#endif
        }
    }
    END_SINGLE_THREAD
#if 1
    if (page) {
        volatile request::Request request;
        request.u.allocOnePageResult.page = page;
        lambda(this, &request);
    }
#endif
}

}  // namespace gloop
#endif  // GLOOP_DEVICE_LOOP_H_
