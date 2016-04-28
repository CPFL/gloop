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
#ifndef GLOOP_HOST_CONTEXT_CU_H_
#define GLOOP_HOST_CONTEXT_CU_H_
#include <cuda.h>
#include <memory>
#include <utility>
#include <vector>
#include <unordered_set>
#include "config.h"
#include "device_context.cuh"
#include "io.cuh"
#include "mapped_memory.cuh"
#include "noncopyable.h"
#include "spinlock.h"
namespace gloop {

class HostLoop;

class HostContext {
GLOOP_NONCOPYABLE(HostContext);
public:
    __host__ ~HostContext();

    __host__ static std::unique_ptr<HostContext> create(HostLoop&, dim3 logicalBlocks, dim3 physicalBlocks = { }, uint32_t pageCount = GLOOP_SHARED_PAGE_COUNT);

    __host__ DeviceContext deviceContext() { return m_context; }

    dim3 blocks() const { return m_logicalBlocks; }

    template<typename Callback>
    __host__ bool tryPeekRequest(const Callback& callback);

    FileDescriptorTable& table() { return m_table; }

    typedef Spinlock Mutex;
    Mutex& mutex() { return m_mutex; }

    void prepareForLaunch();

    uint32_t pending() const;

    void addExitRequired(IPC* ipc)
    {
        // Mutex should be held.
        m_exitRequired.push_back(ipc);
    }

    bool addUnmapRequest(std::shared_ptr<MmapResult> result)
    {
        // Mutex should be held.
        m_unmapRequests.insert(result);
    }

private:
    HostContext(HostLoop& hostLoop, dim3 logicalBlocks, dim3 physicalBlocks, uint32_t pageCount);
    bool initialize(HostLoop&);

    HostLoop& m_hostLoop;
    Mutex m_mutex;
    FileDescriptorTable m_table { };
    std::unique_ptr<IPC[]> m_ipc { nullptr };
    std::shared_ptr<MappedMemory> m_pending { nullptr };
    DeviceContext m_context { nullptr };
    dim3 m_logicalBlocks { };
    dim3 m_physicalBlocks { };
    uint32_t m_pageCount { };
    std::vector<IPC*> m_exitRequired;
    std::unordered_set<std::shared_ptr<MmapResult>> m_unmapRequests;
    bool m_exitHandlerScheduled { false };
};

template<typename Callback>
inline bool HostContext::tryPeekRequest(const Callback& callback)
{
    bool found = false;
    int blocks = m_logicalBlocks.x * m_logicalBlocks.y;
    for (int i = 0; i < blocks; ++i) {
        for (uint32_t j = 0; j < GLOOP_SHARED_SLOT_SIZE; ++j) {
            auto& channel = m_ipc[i * GLOOP_SHARED_SLOT_SIZE + j];
            Code code = channel.peek();
            if (IsOperationCode(code)) {
                found = true;
                callback(&channel);
            }
        }
    }
    return found;
}

}  // namespace gloop
#endif  // GLOOP_HOST_CONTEXT_CU_H_
