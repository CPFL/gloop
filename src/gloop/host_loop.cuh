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
#ifndef GLOOP_HOST_LOOP_H_
#define GLOOP_HOST_LOOP_H_
#include <atomic>
#include <boost/asio.hpp>
#include <boost/interprocess/ipc/message_queue.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/thread.hpp>
#include <deque>
#include <gpufs/libgpufs/fs_initializer.cu.h>
#include <gipc/gipc.cuh>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <thrust/tuple.h>
#include <unordered_map>
#include <uv.h>
#include "command.h"
#include "copy_work_pool.cuh"
#include "copy_worker.cuh"
#include "data_log.h"
#include "entry.cuh"
#include "host_context.cuh"
#include "host_memory.cuh"
#include "ipc.cuh"
#include "make_unique.h"
#include "noncopyable.h"

namespace boost {
class thread;
}  // namespace boost

namespace gloop {

class HostLoop {
GLOOP_NONCOPYABLE(HostLoop);
public:
    ~HostLoop();

    uint32_t id() const { return m_id; }

    void drain();

    static std::unique_ptr<HostLoop> create(int deviceNumber);

    template<typename DeviceLambda, class... Args>
    __host__ void launch(HostContext& context, dim3 threads, const DeviceLambda& callback, Args... args);

private:
    HostLoop(int deviceNumber);

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

    private:
        HostLoop& m_hostLoop;
    };

    void prologue(HostContext&, dim3 threads);
    void epilogue();

    void initialize();

    void runPoller();
    void stopPoller();
    void pollerMain();

    bool handleIO(Command);
    void send(Command);

    void resume();

    void lockLaunch();
    void unlockLaunch();

    void prepareForLaunch();

    int m_deviceNumber { 0 };
    uv_loop_t* m_loop { nullptr };
    std::unique_ptr<boost::thread> m_poller;

    dim3 m_threads { };
    HostContext* m_currentContext { nullptr };

    uint32_t m_id { 0 };
    boost::asio::io_service m_ioService;
    boost::asio::local::stream_protocol::socket m_monitorConnection;
    std::unique_ptr<boost::interprocess::message_queue> m_requestQueue;
    std::unique_ptr<boost::interprocess::message_queue> m_responseQueue;
    std::unique_ptr<boost::interprocess::shared_memory_object> m_sharedMemory;
    std::unique_ptr<boost::interprocess::mapped_region> m_signal;
    volatile uint32_t* m_deviceSignal;
    cudaStream_t m_pgraph;
    CopyWorkPool m_copyWorkPool;
    KernelLock m_kernelLock;
};

template<typename DeviceLambda, class... Args>
inline __host__ void HostLoop::launch(HostContext& hostContext, dim3 threads, const DeviceLambda& callback, Args... args)
{
    prologue(hostContext, threads);
    {
        auto arguments = thrust::make_tuple(std::forward<Args>(args)...);
        std::unique_ptr<boost::asio::io_service::work> kernelWork = make_unique<boost::asio::io_service::work>(m_ioService);
        std::unique_ptr<boost::thread> kernelThread = make_unique<boost::thread>([&] {
            {
                std::lock_guard<KernelLock> lock(m_kernelLock);
                prepareForLaunch();
                while (true) {
                    gloop::launch<<<hostContext.blocks(), m_threads, 0, m_pgraph>>>(m_deviceSignal, hostContext.deviceContext(), callback, arguments);
                    cudaError_t error = cudaGetLastError();
                    if (cudaErrorLaunchOutOfResources == error) {
                        continue;
                    }
                    GLOOP_CUDA_SAFE_CALL(error);
                    break;
                }
                GLOOP_CUDA_SAFE_CALL(cudaStreamSynchronize(m_pgraph));
            }
            while (m_currentContext->pending()) {
                resume();
            }
            kernelWork.reset();
        });
        drain();
    }
    epilogue();
}

inline void HostLoop::lockLaunch()
{
        unsigned int priority { };
        std::size_t size { };
        Command command {
            .type = Command::Type::Lock,
            .payload = 0
        };
        m_requestQueue->send(&command, sizeof(Command), 0);
        m_responseQueue->receive(&command, sizeof(Command), size, priority);
}

inline void HostLoop::unlockLaunch()
{
        Command command {
            .type = Command::Type::Unlock
        };
        m_requestQueue->send(&command, sizeof(Command), 0);
}

}  // namespace gloop
#endif  // GLOOP_HOST_LOOP_H_
