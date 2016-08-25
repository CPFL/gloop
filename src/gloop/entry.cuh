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

#include "device_context.cuh"
#include "device_loop_inlines.cuh"
#include <type_traits>
#include <utility>

namespace gloop {

template <typename DeviceLoop, typename DeviceLambda, class... Args>
inline __device__ void mainLoop(DeviceLoop* loop, int isInitialExecution, DeviceContext& context, const DeviceLambda& callback, Args&&... args)
{
    int callbackKicked = 0;
    BEGIN_SINGLE_THREAD
    {
        if (isInitialExecution) {
            loop->initialize(context);
        } else {
            callbackKicked = loop->initialize(context, DeviceLoop::Resume);
        }
    }
    END_SINGLE_THREAD
    int suspended = 0;
    {
#if defined(GLOOP_ENABLE_ELASTIC_KERNELS)
        if (loop->logicalBlocksCount() == 0)
            return;
#endif

        if (__syncthreads_or(callbackKicked)) {
            suspended = loop->drain();
        }
    }

#if defined(GLOOP_ENABLE_ELASTIC_KERNELS)
    while (__syncthreads_and(!suspended)) {
#endif
        {
            callback(loop, args...);
            suspended = loop->drain();
        }
#if defined(GLOOP_ENABLE_ELASTIC_KERNELS)
    }
#endif
}

template <typename DeviceLambda, class... Args>
inline __global__ void resume(Global, int isInitialExecution, DeviceContext context, const DeviceLambda& callback, Args... args)
{
    DeviceLoop<Global>* loop = reinterpret_cast<DeviceLoop<Global>*>(context.deviceLoopStorage + GLOOP_BID());
    return mainLoop(loop, isInitialExecution, context, callback, std::forward<Args&&>(args)...);
}

template <typename DeviceLambda, class... Args>
inline __global__ void resume(Shared, int isInitialExecution, DeviceContext context, const DeviceLambda& callback, Args... args)
{
    __shared__ DeviceLoop<Shared> sharedDeviceLoop;
    return mainLoop(&sharedDeviceLoop, isInitialExecution, context, callback, std::forward<Args&&>(args)...);
}

} // namespace gloop
