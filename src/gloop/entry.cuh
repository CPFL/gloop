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
#ifndef GLOOP_ENTRY_H_
#define GLOOP_ENTRY_H_
#include <type_traits>
#include <utility>
#include "device_context.cuh"
#include "device_loop_inlines.cuh"

namespace gloop {

template<typename DeviceLambda, class... Args>
inline __global__ void resume(DeviceLoopAllocationPolicyGlobalTag, int isInitialExecution, DeviceContext context, const DeviceLambda& callback, Args... args)
{
    int callbackKicked = 0;
    int suspended = 0;
    DeviceLoop* loop;
    {
        __shared__ DeviceLoop* sharedDeviceLoop;
        __shared__ int sharedCallbackKicked;

        BEGIN_SINGLE_THREAD
        {
            sharedDeviceLoop = new DeviceLoop();
            if (isInitialExecution) {
                sharedCallbackKicked = 0;
                sharedDeviceLoop->initialize(context);
            } else {
                sharedCallbackKicked = sharedDeviceLoop->initialize(context, DeviceLoop::Resume);
            }
        }
        END_SINGLE_THREAD
        callbackKicked = sharedCallbackKicked;
        loop = sharedDeviceLoop;
    }
    {
#if defined(GLOOP_ENABLE_ELASTIC_KERNELS)
        if (loop->logicalBlocksCount() == 0)
            return;
#endif

        if (callbackKicked) {
            suspended = loop->drain();
        }
    }

#if defined(GLOOP_ENABLE_ELASTIC_KERNELS)
    while (!suspended) {
#endif
        {
            callback(loop, args...);
            suspended = loop->drain();
        }
#if defined(GLOOP_ENABLE_ELASTIC_KERNELS)
    }
#endif

    BEGIN_SINGLE_THREAD
    {
        delete loop;
    }
    END_SINGLE_THREAD
}

template<typename DeviceLambda, class... Args>
inline __global__ void resume(DeviceLoopAllocationPolicySharedTag, int isInitialExecution, DeviceContext context, const DeviceLambda& callback, Args... args)
{
    __shared__ DeviceLoop sharedDeviceLoop;
    __shared__ int callbackKicked;
    BEGIN_SINGLE_THREAD
    {
        if (isInitialExecution) {
            callbackKicked = 0;
            sharedDeviceLoop.initialize(context);
        } else {
            callbackKicked = sharedDeviceLoop.initialize(context, DeviceLoop::Resume);
        }
    }
    END_SINGLE_THREAD

#if defined(GLOOP_ENABLE_ELASTIC_KERNELS)
    if (sharedDeviceLoop.logicalBlocksCount() == 0)
        return;
#endif

    int suspended = 0;
    if (callbackKicked) {
        suspended = sharedDeviceLoop.drain();
    }

#if defined(GLOOP_ENABLE_ELASTIC_KERNELS)
    while (!suspended) {
#endif
        {
            callback(&sharedDeviceLoop, args...);
            suspended = sharedDeviceLoop.drain();
        }
#if defined(GLOOP_ENABLE_ELASTIC_KERNELS)
    }
#endif
}

}  // namespace gloop
#endif  // GLOOP_ENTRY_H_
