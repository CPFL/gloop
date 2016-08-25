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

#include "system_initialize.h"
#include "utility.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <mutex>
#include <thread>
#include <unistd.h>

namespace gloop {

void initialize()
{
    // static std::once_flag initializeFlag;
    // std::call_once(initializeFlag, []() {
    // GLOOP_CUDA_SAFE_CALL(cuInit(0));
    // CUdevice device;
    // GLOOP_CUDA_SAFE_CALL(cuDeviceGet(&device, 0));
    // GLOOP_CUDA_SAFE_CALL(cudaDeviceReset());
    // CUcontext primaryContext;
    // GLOOP_CUDA_SAFE_CALL(cuCtxCreate(&primaryContext, CU_CTX_MAP_HOST, device));
    // GLOOP_CUDA_SAFE_CALL(cuCtxSetCurrent(primaryContext));
    // GLOOP_CUDA_SAFE_CALL(cudaSetDeviceFlags(cudaDeviceMapHost | cudaDeviceScheduleSpin));
    // });
}

} // namespace gloop
