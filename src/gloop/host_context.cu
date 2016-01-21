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
#include "device_context.cuh"
#include "ipc.cuh"
#include "host_context.cuh"
#include "make_unique.h"

namespace gloop {

std::unique_ptr<HostContext> HostContext::create(HostLoop& hostLoop, dim3 blocks)
{
    std::unique_ptr<HostContext> hostContext(new HostContext(blocks));
    if (!hostContext->initialize()) {
        return nullptr;
    }
    return hostContext;
}

HostContext::HostContext(dim3 blocks)
    : m_blocks(blocks)
{
}

HostContext::~HostContext()
{
    if (m_context.context) {
        cudaFree(m_context.context);
    }
}

bool HostContext::initialize()
{
    m_ipc = make_unique<IPC[]>(m_blocks.x * m_blocks.y * GLOOP_SHARED_SLOT_SIZE);
    GLOOP_CUDA_SAFE_CALL(cudaHostGetDevicePointer(&m_context.channels, m_ipc.get(), 0));
    GLOOP_CUDA_SAFE_CALL(cudaMalloc(&m_context.context, sizeof(DeviceLoop::PerBlockContext) * m_blocks.x * m_blocks.y));
    GLOOP_CUDA_SAFE_CALL(cudaMalloc(&m_context.pages, sizeof(DeviceLoop::OnePage) * GLOOP_SHARED_PAGE_COUNT * m_blocks.x * m_blocks.y));
    return true;
}

template<typename T, typename U>
T readNoCache(volatile const U* ptr)
{
    return *reinterpret_cast<volatile const T*>(ptr);
}

IPC* HostContext::tryPeekRequest()
{
    IPC* result = nullptr;
    __sync_synchronize();
    int blocks = m_blocks.x * m_blocks.y;
    for (int i = 0; i < blocks; ++i) {
        for (uint32_t j = 0; j < GLOOP_SHARED_SLOT_SIZE; ++j) {
            auto& channel = m_ipc[i * GLOOP_SHARED_SLOT_SIZE + j];
            // printf("channel[%d][%d] = %d\n", (int)i, (int)j, (int)channel.peek());
            Code code = channel.peek();
            if (IsOperationCode(code)) {
                result = &channel;
                break;
            }
        }
    }
    __sync_synchronize();
    return result;
}

}  // namespace gloop
