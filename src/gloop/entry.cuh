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
#include "device_loop.cuh"

namespace gloop {

template<typename Lambda>
inline void tryLaunch(const Lambda& lambda)
{
    while (true) {
        lambda();
        cudaError_t error = cudaGetLastError();
        if (cudaErrorLaunchOutOfResources == error) {
            continue;
        }
        GLOOP_CUDA_SAFE_CALL(error);
        break;
    }
}

typedef std::aligned_storage<sizeof(DeviceLoop), alignof(DeviceLoop)>::type UninitializedDeviceLoopStorage;

template<typename DeviceLambda, class... Args>
inline __global__ void resume(volatile uint32_t* signal, DeviceContext context, const DeviceLambda& callback, Args... args)
{
    __shared__ int callbackKicked;
    BEGIN_SINGLE_THREAD
    {
        if (signal) {
            callbackKicked = 0;
            sharedDeviceLoop.initialize(signal, context);
        } else {
            callbackKicked = sharedDeviceLoop.initialize(signal, context, DeviceLoop::Resume);
        }
    }
    END_SINGLE_THREAD

    if (sharedDeviceLoop.logicalBlocksCount() == 0)
        return;

    // __threadfence_system();
    int suspended = 0;

    do {
        if (!callbackKicked) {
            callback(&sharedDeviceLoop, args...);
        } else {
            BEGIN_SINGLE_THREAD
            {
                callbackKicked = 0;
            }
            END_SINGLE_THREAD
        }
        suspended = sharedDeviceLoop.drain();
    } while (!suspended);
    __threadfence_system();  // FIXME
}

}  // namespace gloop
#endif  // GLOOP_ENTRY_H_
