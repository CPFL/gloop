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
#include <boost/interprocess/sync/named_mutex.hpp>
#include <deque>
#include <gpufs/libgpufs/fs_initializer.cu.h>
#include <gipc/gipc.cuh>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <uv.h>
#include "command.h"
#include "entry.cuh"
#include "host_context.cuh"
#include "host_memory.cuh"
#include "ipc.cuh"
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

    void initialize();

    void runPoller();
    void stopPoller();
    void pollerMain();

    bool handle(Command);
    void send(Command);

    bool hostBack();

    void resume();
    void registerKernelCompletionCallback(cudaStream_t);

    int m_deviceNumber;
    uv_loop_t* m_loop;
    std::atomic<bool> m_stop { false };
    std::atomic<bool> m_hostBack { false };
    std::unique_ptr<boost::thread> m_poller;

    dim3 m_threads { };
    HostContext* m_currentContext { nullptr };

    std::unique_ptr<IPC> m_channel;

    uint32_t m_id { 0 };
    boost::asio::io_service m_ioService;
    boost::asio::local::stream_protocol::socket m_socket;
    std::unique_ptr<boost::interprocess::message_queue> m_requestQueue;
    std::unique_ptr<boost::interprocess::message_queue> m_responseQueue;
    std::unique_ptr<boost::interprocess::named_mutex> m_launchMutex;
    std::unordered_map<std::string, File> m_fds { };
    cudaStream_t m_pgraph;
    cudaStream_t m_pcopy0;
    cudaStream_t m_pcopy1;
    std::deque<std::shared_ptr<HostMemory>> m_pool;
};

template<typename DeviceLambda, class... Args>
inline __host__ void HostLoop::launch(HostContext& hostContext, dim3 threads, const DeviceLambda& callback, Args... args)
{
    m_threads = threads;
    m_currentContext = &hostContext;
    {
        m_launchMutex->lock();
        m_currentContext->prepareForLaunch();
        while (true) {
            gloop::launch<<<hostContext.blocks(), m_threads, 0, m_pgraph>>>(hostContext.deviceContext(), callback, std::forward<Args>(args)...);
            cudaError_t error = cudaGetLastError();
            if (cudaErrorLaunchOutOfResources == error) {
                continue;
            }
            GLOOP_CUDA_SAFE_CALL(error);
            break;
        }
        registerKernelCompletionCallback(m_pgraph);
    }
    drain();
    m_currentContext = nullptr;
}

}  // namespace gloop
#endif  // GLOOP_HOST_LOOP_H_
