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
#include <boost/thread.hpp>
#include <cuda.h>
#include <memory>
#include <mutex>
#include <unordered_set>
#include <utility>
#include <vector>
#include "config.h"
#include "device_context.cuh"
#include "rpc.cuh"
#include "io.cuh"
#include "mapped_memory.cuh"
#include "noncopyable.h"
#include "spinlock.h"
#include "utility.cuh"
namespace gloop {

class HostLoop;
struct RPC;

class HostContext {
GLOOP_NONCOPYABLE(HostContext);
public:
    friend struct RPC;

    __host__ ~HostContext();

    __host__ static std::unique_ptr<HostContext> create(HostLoop&, dim3 maxPhysicalBlocks, uint32_t pageCount = GLOOP_SHARED_PAGE_COUNT);

    __host__ DeviceContext deviceContext() { return m_context; }

    dim3 maxPhysicalBlocks() const { return m_maxPhysicalBlocks; }
    dim3 physicalBlocks() const { return m_physicalBlocks; }

    template<typename Callback>
    __host__ bool tryPeekRequest(Callback callback);

    FileDescriptorTable& table() { return m_table; }

    typedef Spinlock Mutex;
    typedef boost::condition_variable_any Condition;
    Mutex& mutex() { return m_mutex; }
    Condition& condition() { return m_condition; }


    void prepareForLaunch();

    uint32_t pending() const;

    void addExitRequired(const std::lock_guard<Mutex>&, RPC rpc)
    {
        // Mutex should be held.
        m_exitRequired.push_back(rpc);
    }

    bool addUnmapRequest(const std::lock_guard<Mutex>&, std::shared_ptr<MmapResult> result)
    {
        // Mutex should be held.
        m_unmapRequests.insert(result);
    }

    bool isReadyForResume(const std::unique_lock<Mutex>&);

    void prologue(dim3 logicalBlocks, dim3 physicalBlocks);
    void epilogue();

private:
    HostContext(HostLoop& hostLoop, dim3 maxPhysicalBlocks, uint32_t pageCount);
    bool initialize(HostLoop&);

    template<typename Callback>
    void forEachRPC(Callback callback);

    void pollerMain();

    HostLoop& m_hostLoop;
    Mutex m_mutex { };
    Condition m_condition { };
    FileDescriptorTable m_table { };
    std::shared_ptr<MappedMemory> m_codesMemory { nullptr };
    std::shared_ptr<MappedMemory> m_payloadsMemory { nullptr };
    std::shared_ptr<MappedMemory> m_kernel { nullptr };
    std::shared_ptr<MappedMemory> m_hostContextMemory { nullptr };
    DeviceContext m_context { nullptr };
    int32_t* m_codes;
    DeviceContext::PerBlockHostContext* m_hostContext { nullptr };
    request::Payload* m_payloads;
    dim3 m_logicalBlocks { };
    dim3 m_maxPhysicalBlocks { };
    dim3 m_physicalBlocks { };
    uint32_t m_pageCount { };
    std::vector<RPC> m_exitRequired;
    std::unordered_set<std::shared_ptr<MmapResult>> m_unmapRequests;
    bool m_exitHandlerScheduled { false };
    std::unique_ptr<boost::thread> m_poller;
};

GLOOP_ALWAYS_INLINE __host__ void RPC::emit(HostContext& hostContext, Code code)
{
    syncWrite(&hostContext.m_codes[position], static_cast<int32_t>(code));
}

GLOOP_ALWAYS_INLINE __host__ Code RPC::peek(HostContext& hostContext)
{
    return readNoCache<Code>(&hostContext.m_codes[position]);
}

GLOOP_ALWAYS_INLINE __host__ request::Payload* RPC::request(HostContext& hostContext) const
{
    return &hostContext.m_payloads[position];
}

template<typename Callback>
inline void HostContext::forEachRPC(Callback callback)
{
    uint64_t blocks = sumOfBlocks(m_physicalBlocks);
    for (uint64_t i = 0; i < blocks; ++i) {
        for (uint32_t j = 0; j < GLOOP_SHARED_SLOT_SIZE; ++j) {
            RPC rpc { i * GLOOP_SHARED_SLOT_SIZE + j };
            callback(rpc);
        }
    }
}

template<typename Callback>
inline bool HostContext::tryPeekRequest(Callback callback)
{
    bool found = false;
    forEachRPC([&](RPC rpc) {
        Code code = rpc.peek(*this);
        if (IsOperationCode(code)) {
            found = true;
            callback(rpc);
        }
    });
    return found;
}

}  // namespace gloop
#endif  // GLOOP_HOST_CONTEXT_CU_H_
