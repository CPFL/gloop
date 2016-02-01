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
#include <boost/asio.hpp>
#include <boost/thread.hpp>
#include <cassert>
#include <cstdio>
#include <cuda.h>
#include <gpufs/libgpufs/fs_initializer.cu.h>
#include <gpufs/libgpufs/host_loop.h>
#include <memory>
#include "bitwise_cast.h"
#include "command.h"
#include "config.h"
#include "data_log.h"
#include "helper.cuh"
#include "host_loop.cuh"
#include "io.cuh"
#include "ipc.cuh"
#include "make_unique.h"
#include "memcpy_io.cuh"
#include "monitor_session.h"
#include "request.h"
#include "system_initialize.h"
#include "utility.h"
namespace gloop {

__device__ IPC* g_channel;

HostLoop::HostLoop(int deviceNumber)
    : m_deviceNumber(deviceNumber)
    , m_loop(uv_loop_new())
    , m_socket(m_ioService)
{
    // Connect to the gloop monitor.
    {
        m_socket.connect(boost::asio::local::stream_protocol::endpoint(GLOOP_ENDPOINT));
        Command command = {
            .type = Command::Type::Initialize,
        };
        Command result { };
        while (true) {
            boost::system::error_code error;
            boost::asio::write(
                m_socket,
                boost::asio::buffer(reinterpret_cast<const char*>(&command), sizeof(Command)),
                boost::asio::transfer_all(),
                error);
            if (error != boost::asio::error::make_error_code(boost::asio::error::interrupted)) {
                break;
            }
            // retry
        }
        while (true) {
            boost::system::error_code error;
            boost::asio::read(
                m_socket,
                boost::asio::buffer(reinterpret_cast<char*>(&result), sizeof(Command)),
                boost::asio::transfer_all(),
                error);
            if (error != boost::asio::error::make_error_code(boost::asio::error::interrupted)) {
                break;
            }
        }
        m_id = result.payload;
    }
    m_requestQueue = monitor::Session::createQueue(GLOOP_SHARED_REQUEST_QUEUE, m_id, false);
    m_responseQueue = monitor::Session::createQueue(GLOOP_SHARED_RESPONSE_QUEUE, m_id, false);
    m_launchMutex = monitor::Session::createMutex(GLOOP_SHARED_LAUNCH_MUTEX, m_id, false);
    GLOOP_DEBUG("id:(%u)\n", m_id);
}

HostLoop::~HostLoop()
{
    uv_loop_close(m_loop);
    stopPoller();
}

std::unique_ptr<HostLoop> HostLoop::create(int deviceNumber)
{
    gloop::initialize();
    std::unique_ptr<HostLoop> hostLoop(new HostLoop(deviceNumber));
    hostLoop->initialize();
    return hostLoop;
}

// GPU RPC poller.
void HostLoop::runPoller()
{
    assert(!m_poller);
    m_stop.store(false, std::memory_order_release);
    m_poller = make_unique<boost::thread>([this]() {
        pollerMain();
    });
}

void HostLoop::stopPoller()
{
    m_stop.store(true, std::memory_order_release);
    if (m_poller) {
        m_poller->join();
        m_poller.reset();
    }
}

void HostLoop::send(Command command)
{
    m_responseQueue->send(&command, sizeof(Command), 0);
}

void HostLoop::pollerMain()
{
    while (true) {
        if (!m_currentContext)
            continue;

        if (IPC* ipc = m_currentContext->tryPeekRequest()) {
            request::Request req { };
            memcpyIO(&req, ipc->request(), sizeof(request::Request));
            ipc->emit(Code::None);
            send({
                .type = Command::Type::IO,
                .payload = bitwise_cast<uintptr_t>(ipc),
                .request = req,
            });
            continue;
        }

        if (m_stop.load(std::memory_order_acquire)) {
            break;
        }
    }
}

__global__ static void initializeDevice(IPC* channel)
{
    g_channel = channel;
}

void HostLoop::initialize()
{
    {
        std::lock_guard<decltype(*m_launchMutex)> lock(*m_launchMutex);
        // This ensures that primary GPU context is initialized.
        GLOOP_CUDA_SAFE_CALL(cudaStreamCreate(&m_pgraph));
        GLOOP_CUDA_SAFE_CALL(cudaStreamCreate(&m_pcopy0));
        GLOOP_CUDA_SAFE_CALL(cudaStreamCreate(&m_pcopy1));
    }

    m_channel = make_unique<IPC>();

    IPC* deviceChannel = nullptr;
    GLOOP_CUDA_SAFE_CALL(cudaHostGetDevicePointer(&deviceChannel, m_channel.get(), 0));

    for (int i = 0; i < 16; ++i) {
        m_pool.push_back(HostMemory::create(GLOOP_SHARED_PAGE_SIZE, cudaHostAllocPortable));
    }

	cudaThreadSynchronize();
	CUDA_SAFE_CALL(cudaPeekAtLastError());
}

void HostLoop::registerKernelCompletionCallback(cudaStream_t stream)
{
    GLOOP_CUDA_SAFE_CALL(cudaStreamAddCallback(stream, [](cudaStream_t stream, cudaError_t error, void* userData) {
        HostLoop* hostLoop = static_cast<HostLoop*>(userData);
        hostLoop->m_launchMutex->unlock();
        hostLoop->send({
            .type = Command::Type::Operation,
            .payload = Command::Operation::Complete,
        });
    }, this, /* Must be 0. */ 0));
}

void HostLoop::drain()
{
    runPoller();
    while (true) {
        Command result = { };
        unsigned int priority { };
        std::size_t size { };
        m_responseQueue->receive(&result, sizeof(Command), size, priority);
        if (handle(result)) {
            break;
        }
    }
    stopPoller();
    logGPUfsDone();
}

bool HostLoop::hostBack()
{
    m_stop.store(true, std::memory_order_release);
    while (m_stop.load(std::memory_order_acquire));
    return true;
}

void HostLoop::resume()
{
    m_launchMutex->lock();
    m_currentContext->prepareForLaunch();
    tryLaunch([&] {
        gloop::resume<<<m_currentContext->blocks(), m_threads, 0, m_pgraph>>>(m_currentContext->deviceContext());
    });
    registerKernelCompletionCallback(m_pgraph);
}

bool HostLoop::handle(Command command)
{
    switch (command.type) {
    case Command::Type::Initialize: {
        GLOOP_UNREACHABLE();
        break;
    }

    case Command::Type::Operation: {
        switch (static_cast<Command::Operation>(command.payload)) {
        case Command::Operation::HostBack:
            return hostBack();

        case Command::Operation::DeviceLoopComplete:
            return true;

        case Command::Operation::Complete:
            // Still pending callbacks are queued.
            // GLOOP_DEBUG("resume %u\n", m_currentContext->pending());
            GLOOP_CUDA_SAFE_CALL(cudaStreamSynchronize(m_pgraph));
            __sync_synchronize();
            if (m_currentContext->pending()) {
                resume();
                return false;
            }
            // There is no callbacks.
            return true;
        }
        break;
    }

    case Command::Type::IO: {
        const request::Request& req = command.request;
        IPC* ipc = bitwise_cast<IPC*>(command.payload);

        switch (static_cast<Code>(req.code)) {
        case Code::Open: {
            int fd = m_currentContext->table().open(req.u.open.filename.data, req.u.open.mode);
            GLOOP_DEBUG("Open %s %d\n", req.u.open.filename.data, fd);
            ipc->request()->u.openResult.fd = fd;
            ipc->emit(Code::Complete);
            break;
        }

        case Code::Write: {
            // FIXME: Significant naive implementaion.
            // We should integrate implementation with GPUfs's buffer cache.
            GLOOP_DEBUG("Write fd:(%d),count:(%u),offset:(%d),page:(%p)\n", req.u.write.fd, (unsigned)req.u.write.count, (int)req.u.write.offset, (void*)req.u.read.buffer);

            std::shared_ptr<HostMemory> hostMemory = m_pool.front();
            m_pool.pop_front();
            assert(req.u.write.count <= hostMemory->size());

            cudaStream_t stream = m_pcopy0;
            GLOOP_CUDA_SAFE_CALL(cudaMemcpyAsync(hostMemory->hostPointer(), req.u.write.buffer, req.u.write.count, cudaMemcpyDeviceToHost, stream));
            GLOOP_CUDA_SAFE_CALL(cudaStreamSynchronize(stream));
            __sync_synchronize();

            ssize_t writtenCount = pwrite(req.u.write.fd, hostMemory->hostPointer(), req.u.write.count, req.u.write.offset);

            m_pool.push_back(hostMemory);

            ipc->request()->u.writeResult.writtenCount = writtenCount;
            ipc->emit(Code::Complete);
            break;
        }

        case Code::Read: {
            // FIXME: Significant naive implementaion.
            // We should integrate implementation with GPUfs's buffer cache.
            GLOOP_DEBUG("Read ipc:(%p),fd:(%d),count:(%u),offset(%d),page:(%p)\n", (void*)ipc, req.u.read.fd, (unsigned)req.u.read.count, (int)req.u.read.offset, (void*)req.u.read.buffer);

            std::shared_ptr<HostMemory> hostMemory = m_pool.front();
            m_pool.pop_front();
            assert(req.u.read.count <= hostMemory->size());
            ssize_t readCount = pread(req.u.read.fd, hostMemory->hostPointer(), req.u.read.count, req.u.read.offset);
            __sync_synchronize();

            // FIXME: Should use multiple streams. And execute async.
            cudaStream_t stream = m_pcopy1;
            GLOOP_CUDA_SAFE_CALL(cudaMemcpyAsync(req.u.read.buffer, hostMemory->hostPointer(), readCount, cudaMemcpyHostToDevice, stream));
            GLOOP_CUDA_SAFE_CALL(cudaStreamSynchronize(stream));

            m_pool.push_back(hostMemory);

            ipc->request()->u.readResult.readCount = readCount;
            ipc->emit(Code::Complete);
            break;
        }

        case Code::Fstat: {
            struct stat buf { };
            ::fstat(req.u.fstat.fd, &buf);
            GLOOP_DEBUG("Fstat %d %u\n", req.u.fstat.fd, buf.st_size);
            ipc->request()->u.fstatResult.size = buf.st_size;
            ipc->emit(Code::Complete);
            break;
        }

        case Code::Close: {
            m_currentContext->table().close(req.u.close.fd);
            GLOOP_DEBUG("Close %d\n", req.u.close.fd);
            ipc->request()->u.closeResult.error = 0;
            ipc->emit(Code::Complete);
            break;
        }
        }
        return false;
    }
    }
    return false;
}

}  // namespace gloop
