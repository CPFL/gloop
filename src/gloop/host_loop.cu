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
    bool done = false;
    while (!done) {
        // if (m_stop.load(std::memory_order_acquire)) {
        //     m_channel->stop();
        //     while (cudaErrorNotReady != cudaStreamQuery(streamMgr->kernelStream));
        //     m_stop.store(false, std::memory_order_acquire);
        //     break;
        // }

        if (IPC* ipc = m_currentContext->tryPeekRequest()) {
            request::Request req { };
            memcpyIO(&req, ipc->request(), sizeof(request::Request));
            ipc->emit(Code::None);
            send({
                .type = Command::Type::IO,
                .payload = bitwise_cast<uintptr_t>(ipc),
                .request = req,
            });
        }

        // open_loop(this, m_deviceNumber);
        // rw_loop(this);
        if (cudaErrorNotReady != cudaStreamQuery(streamMgr->kernelStream)) {
            logGPUfsDone();
            break;
        }
        // async_close_loop(this);
    }
    send({
        .type = Command::Type::Operation,
        .payload = Command::Operation::Complete,
    });
}

__global__ static void initializeDevice(IPC* channel)
{
    g_channel = channel;
}

void HostLoop::initialize()
{
    m_channel = make_unique<IPC>();
    // this must be done from a single thread!
	init_fs<<<1,1>>>(
            cpu_ipcOpenQueue,
            cpu_ipcRWQueue,
            ipcRWManager,
            otable,
            ppool,
            rawStorage,
            ftable,
			rtree_pool,
		 	rtree_array,
			_preclose_table);

    // FIXME: This is not efficient.
    IPC* deviceChannel = nullptr;
    GLOOP_CUDA_SAFE_CALL(cudaHostGetDevicePointer(&deviceChannel, m_channel.get(), 0));
    initializeDevice<<<1, 1>>>(deviceChannel);

	cudaThreadSynchronize();
	CUDA_SAFE_CALL(cudaPeekAtLastError());
}

void HostLoop::wait()
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
}

bool HostLoop::hostBack()
{
    m_stop.store(true, std::memory_order_release);
    while (m_stop.load(std::memory_order_acquire));
    return true;
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
            printf("Open %s %d\n", req.u.open.filename.data, fd);
            ipc->request()->u.result.result = fd;
            ipc->emit(Code::Complete);
            break;
        }

        case Code::Write: {
            break;
        }

        case Code::Read: {
            break;
        }

        case Code::Fstat: {
            struct stat buf { };
            ::fstat(req.u.fstat.fd, &buf);
            printf("Fstat %d %u\n", req.u.fstat.fd, buf.st_size);
            ipc->request()->u.result.result = buf.st_size;
            ipc->emit(Code::Complete);
            break;
        }

        case Code::Close: {
            m_currentContext->table().close(req.u.close.fd);
            ipc->request()->u.result.result = 0;
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
