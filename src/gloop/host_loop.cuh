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
#ifndef GLOOP_HOST_LOOP_H_
#define GLOOP_HOST_LOOP_H_
#include <atomic>
#include <boost/asio.hpp>
#include <boost/interprocess/ipc/message_queue.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/thread.hpp>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include "benchmark.h"
#include "command.h"
#include "rpc.cuh"
#include "noncopyable.h"

namespace boost {
class thread;
}  // namespace boost

namespace gloop {

class CopyWork;
class CopyWorkPool;

class HostLoop {
GLOOP_NONCOPYABLE(HostLoop);
public:
    ~HostLoop();

    uint32_t id() const { return m_id; }

    uint64_t killClock() const { return m_deviceProperties.clockRate * GLOOP_KILL_TIME / 1000; }

    cudaStream_t pgraph() const { return m_pgraph; }

    static std::unique_ptr<HostLoop> create(int deviceNumber, uint64_t costPerBit = 1);

    // Default shared policy launch interfaces.
    template<typename DeviceLambda, class... Args>
    inline __host__ void launch(HostContext& context, dim3 logicalBlocks, dim3 threads, DeviceLambda&& callback, Args&&... args);

    template<typename DeviceLambda, class... Args>
    inline __host__ void launch(HostContext& context, dim3 physicalBlocks, dim3 logicalBlocks, dim3 threads, DeviceLambda&& callback, Args&&... args);

    template<typename Policy, typename DeviceLambda, class... Args>
    inline __host__ void launch(HostContext& context, dim3 logicalBlocks, dim3 threads, DeviceLambda&& callback, Args&&... args);

    template<typename Policy, typename DeviceLambda, class... Args>
    inline __host__ void launch(HostContext& context, dim3 physicalBlocks, dim3 logicalBlocks, dim3 threads, DeviceLambda&& callback, Args&&... args);

    // Generic interfaces.
    template<typename Policy, typename DeviceLambda, class... Args>
    inline __host__ void launchWithSharedMemory(HostContext& context, dim3 logicalBlocks, dim3 threads, size_t sharedMemorySize, DeviceLambda&& callback, Args&&... args);

    template<typename Policy, typename DeviceLambda, class... Args>
    inline __host__ void launchWithSharedMemory(HostContext& context, dim3 physicalBlocks, dim3 logicalBlocks, dim3 threads, size_t sharedMemorySize, DeviceLambda&& callback, Args&&... args);

    class KernelLock {
    GLOOP_NONCOPYABLE(KernelLock);
    public:
        KernelLock(HostLoop& hostLoop) : m_hostLoop(hostLoop) { }

        void lock()
        {
            m_hostLoop.lockLaunch();
        }

        void unlock()
        {
            m_hostLoop.unlockLaunch();
        }

        void unlock(Command::ReleaseStatus status)
        {
            m_hostLoop.unlockLaunch(status);
        }

    private:
        HostLoop& m_hostLoop;
    };

    KernelLock& kernelLock() { return m_kernelLock; }

    bool handleIO(HostContext& context, RPC, Code code, request::Request);

    // Per thread initialization.
    void initializeInThread();

private:
    HostLoop(int deviceNumber, uint64_t costPerBit);

    void refKernel();
    void derefKernel();

    // System initialization.
    void initialize();


    void send(Command);

    template<typename Policy, typename DeviceLambda, class... Args>
    inline __host__ void launchInternal(HostContext& context, dim3 physicalBlocks, dim3 logicalBlocks, dim3 threads, size_t sharedMemorySize, DeviceLambda callback, Args... args);

    template<typename Policy, typename DeviceLambda, typename... Args>
    inline void resume(HostContext&, dim3 threads, DeviceLambda callback, Args... args);

    inline void lockLaunch();
    inline void unlockLaunch(Command::ReleaseStatus = Command::ReleaseStatus::IO);

    void prepareForLaunch(HostContext&);

    void drain();

    CopyWork* acquireCopyWork();
    void releaseCopyWork(CopyWork* copyWork);

    bool threadReady();

    int m_deviceNumber { 0 };

    uint32_t m_id { 0 };
    boost::asio::io_service m_ioService;
    boost::asio::io_service m_kernelService;
    boost::asio::local::stream_protocol::socket m_monitorConnection;
    std::unique_ptr<boost::interprocess::message_queue> m_requestQueue;
    std::unique_ptr<boost::interprocess::message_queue> m_responseQueue;
    std::unique_ptr<boost::interprocess::shared_memory_object> m_sharedMemory;
    std::unique_ptr<boost::interprocess::mapped_region> m_signal;
    volatile uint32_t* m_deviceSignal;
    cudaStream_t m_pgraph;
    std::unique_ptr<CopyWorkPool> m_copyWorkPool;
    KernelLock m_kernelLock;
    cudaDeviceProp m_deviceProperties;

    // Thread group management.
    boost::thread_group m_threadGroup;
    boost::mutex m_threadGroupMutex { };
    boost::condition_variable m_threadGroupNotify;
    boost::condition_variable m_threadGroupReadyNotify;
    int m_threadGroupReadyCount { 0 };
    bool m_stopThreadGroup { false };

    std::unique_ptr<boost::asio::io_service::work> m_kernelWork;
    boost::condition_variable m_ioCompletionNotify;
};

}  // namespace gloop
#endif  // GLOOP_HOST_LOOP_H_
