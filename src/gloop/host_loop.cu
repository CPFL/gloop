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
#include <boost/bind.hpp>
#include <boost/thread.hpp>
#include <cassert>
#include <cstdio>
#include <cuda.h>
#include <gpufs/libgpufs/fs_initializer.cu.h>
#include <gpufs/libgpufs/host_loop.h>
#include <memory>
#include <sys/mman.h>
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
#include "sync_read_write.h"
#include "system_initialize.h"
#include "utility.h"
namespace gloop {

__device__ IPC* g_channel;

HostLoop::HostLoop(int deviceNumber)
    : m_deviceNumber(deviceNumber)
    , m_loop(uv_loop_new())
    , m_ioService()
    , m_monitorConnection(m_ioService)
    , m_kernelLock(*this)
{
    // Connect to the gloop monitor.
    {
        m_monitorConnection.connect(boost::asio::local::stream_protocol::endpoint(GLOOP_ENDPOINT));
        Command command = {
            .type = Command::Type::Initialize,
        };
        Command result { };
        while (true) {
            boost::system::error_code error;
            boost::asio::write(
                m_monitorConnection,
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
                m_monitorConnection,
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
    m_sharedMemory = monitor::Session::createMemory(GLOOP_SHARED_MEMORY, m_id, GLOOP_SHARED_MEMORY_SIZE, false);
    m_signal = make_unique<boost::interprocess::mapped_region>(*m_sharedMemory.get(), boost::interprocess::read_write, /* Offset. */ 0, GLOOP_SHARED_MEMORY_SIZE);
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
    m_poller = make_unique<boost::thread>([this]() {
        pollerMain();
    });
}

void HostLoop::stopPoller()
{
    if (m_poller) {
        m_poller->interrupt();
        m_poller->join();
        m_poller.reset();
    }
}

void HostLoop::pollerMain()
{
    while (true) {
        if (m_currentContext) {
            if (IPC* ipc = m_currentContext->tryPeekRequest()) {
                request::Request req { };
                memcpyIO(&req, ipc->request(), sizeof(request::Request));
                ipc->emit(Code::None);
                handleIO({
                    .type = Command::Type::IO,
                    .payload = bitwise_cast<uintptr_t>(ipc),
                    .request = req,
                });
                continue;
            }
        }
        boost::this_thread::interruption_point();
    }
}

void HostLoop::initialize()
{
    {
        // This ensures that primary GPU context is initialized.
        std::lock_guard<KernelLock> lock(m_kernelLock);
        GLOOP_CUDA_SAFE_CALL(cudaStreamCreate(&m_pgraph));

        GLOOP_CUDA_SAFE_CALL(cudaHostRegister(m_signal->get_address(), GLOOP_SHARED_MEMORY_SIZE, cudaHostRegisterMapped));
        GLOOP_CUDA_SAFE_CALL(cudaHostGetDevicePointer(&m_deviceSignal, m_signal->get_address(), 0));

        for (int i = 0; i < GLOOP_THREAD_GROUP_SIZE; ++i) {
            m_copyWorkPool.release(CopyWork::create());
        }

        CUDA_SAFE_CALL(cudaPeekAtLastError());
    }
}

void HostLoop::drain()
{
    // Host main loop.
#if 0
    // Run in main thread.
    m_ioService.run();
#else
    // Since kernel work is already held by kernel executing thread,
    // when joining threads, we can say that all the events produced by ASIO
    // is already drained.
    boost::thread_group threadGroup;
    for (int i = 0; i < GLOOP_THREAD_GROUP_SIZE; ++i) {
        threadGroup.create_thread(boost::bind(&boost::asio::io_service::run, &m_ioService));
    }
    threadGroup.join_all();
#endif
}

void HostLoop::prepareForLaunch()
{
    m_currentContext->prepareForLaunch();
    syncWrite<uint32_t>(static_cast<volatile uint32_t*>(m_signal->get_address()), 0);
}

void HostLoop::resume()
{
    std::lock_guard<KernelLock> lock(m_kernelLock);
    prepareForLaunch();
    tryLaunch([&] {
        gloop::resume<<<m_currentContext->blocks(), m_threads, 0, m_pgraph>>>(m_deviceSignal, m_currentContext->deviceContext());
    });
    GLOOP_CUDA_SAFE_CALL(cudaStreamSynchronize(m_pgraph));
}

void HostLoop::prologue(HostContext& hostContext, dim3 threads)
{
    m_threads = threads;
    m_currentContext = &hostContext;
    runPoller();
}

void HostLoop::epilogue()
{
    stopPoller();
    logGPUfsDone();
    m_currentContext = nullptr;
}

bool HostLoop::handleIO(Command command)
{
    assert(command.type == Command::Type::IO);
    request::Request req = command.request;
    IPC* ipc = bitwise_cast<IPC*>(command.payload);

    switch (static_cast<Code>(req.code)) {
    case Code::Open: {
        int fd = m_currentContext->table().open(req.u.open.filename.data, req.u.open.mode);
        // GLOOP_DEBUG("open:(%s),fd:(%d)\n", req.u.open.filename.data, fd);
        ipc->request()->u.openResult.fd = fd;
        ipc->emit(Code::Complete);
        break;
    }

    case Code::Write: {
        // FIXME: Significant naive implementaion.
        // We should integrate implementation with GPUfs's buffer cache.
        m_ioService.post([ipc, req, this]() {
            // GLOOP_DEBUG("Write fd:(%d),count:(%u),offset:(%d),page:(%p)\n", req.u.write.fd, (unsigned)req.u.write.count, (int)req.u.write.offset, (void*)req.u.read.buffer);
            std::shared_ptr<CopyWork> copyWork = m_copyWorkPool.acquire();
            assert(req.u.write.count <= copyWork->hostMemory().size());

            GLOOP_CUDA_SAFE_CALL(cudaMemcpyAsync(copyWork->hostMemory().hostPointer(), req.u.write.buffer, req.u.write.count, cudaMemcpyDeviceToHost, copyWork->stream()));
            GLOOP_CUDA_SAFE_CALL(cudaStreamSynchronize(copyWork->stream()));
            __sync_synchronize();

            ssize_t writtenCount = ::pwrite(req.u.write.fd, copyWork->hostMemory().hostPointer(), req.u.write.count, req.u.write.offset);

            m_copyWorkPool.release(copyWork);

            ipc->request()->u.writeResult.writtenCount = writtenCount;
            ipc->emit(Code::Complete);
        });
        break;
    }

    case Code::Read: {
        // FIXME: Significant naive implementaion.
        // We should integrate implementation with GPUfs's buffer cache.
        m_ioService.post([ipc, req, this]() {
            // GLOOP_DEBUG("Read ipc:(%p),fd:(%d),count:(%u),offset(%d),page:(%p)\n", (void*)ipc, req.u.read.fd, (unsigned)req.u.read.count, (int)req.u.read.offset, (void*)req.u.read.buffer);

            std::shared_ptr<CopyWork> copyWork = m_copyWorkPool.acquire();
            assert(req.u.read.count <= copyWork->hostMemory().size());
            ssize_t readCount = ::pread(req.u.read.fd, copyWork->hostMemory().hostPointer(), req.u.read.count, req.u.read.offset);
            __sync_synchronize();

            // FIXME: Should use multiple streams. And execute async.
            GLOOP_CUDA_SAFE_CALL(cudaMemcpyAsync(req.u.read.buffer, copyWork->hostMemory().hostPointer(), readCount, cudaMemcpyHostToDevice, copyWork->stream()));
            GLOOP_CUDA_SAFE_CALL(cudaStreamSynchronize(copyWork->stream()));

            m_copyWorkPool.release(copyWork);

            ipc->request()->u.readResult.readCount = readCount;
            ipc->emit(Code::Complete);
        });
        break;
    }

    case Code::Fstat: {
        struct stat buf { };
        ::fstat(req.u.fstat.fd, &buf);
        // GLOOP_DEBUG("Fstat %d %u\n", req.u.fstat.fd, buf.st_size);
        ipc->request()->u.fstatResult.size = buf.st_size;
        ipc->emit(Code::Complete);
        break;
    }

    case Code::Close: {
        m_currentContext->table().close(req.u.close.fd);
        // GLOOP_DEBUG("Close %d\n", req.u.close.fd);
        ipc->request()->u.closeResult.error = 0;
        ipc->emit(Code::Complete);
        break;
    }

    case Code::Mmap: {
        // FIXME: Significant naive implementaion.
        // We should integrate implementation with GPUfs's buffer cache.
        m_ioService.post([ipc, req, this]() {
            // void* host = ::mmap(req.u.mmap.address, req.u.mmap.size, req.u.mmap.prot, req.u.mmap.flags, req.u.mmap.fd, req.u.mmap.offset);
            // void* host = ::mmap(req.u.mmap.address, req.u.mmap.size, req.u.mmap.prot, req.u.mmap.flags, MAP_ANONYMOUS, req.u.mmap.offset);
            void* host = ::mmap(req.u.mmap.address, req.u.mmap.size, req.u.mmap.prot, req.u.mmap.flags, req.u.mmap.fd, req.u.mmap.offset);
            GLOOP_DEBUG("mmap:address(%p),size:(%u),prot:(%d),flags:(%d),fd:(%d),offset:(%d),res:(%p)\n", req.u.mmap.address, req.u.mmap.size, req.u.mmap.prot, req.u.mmap.flags, req.u.mmap.fd, req.u.mmap.offset, host);
            void* device = nullptr;
            // volatile uint32_t value = *((volatile uint32_t*)host);
            // *((volatile uint32_t*)host) = value;
            GLOOP_CUDA_SAFE_CALL(cudaHostRegister(host, req.u.mmap.size, cudaHostRegisterMapped));
            GLOOP_CUDA_SAFE_CALL(cudaHostGetDevicePointer(&device, host, 0));
            {
                std::lock_guard<HostContext::Mutex> guard(m_currentContext->mutex());
                m_currentContext->table().registerMapping(host, device);
                ipc->request()->u.mmapResult.address = device;
                ipc->emit(Code::ExitRequired);
                m_currentContext->addExitRequired(ipc);
            }
        });
        break;
    }

    case Code::Munmap: {
        // FIXME: Significant naive implementaion.
        // We should integrate implementation with GPUfs's buffer cache.
        m_ioService.post([ipc, req, this]() {
            GLOOP_DEBUG("munmap:address(%p),size:(%u)\n", req.u.munmap.address, req.u.munmap.size);
            // FIXME: We should schedule this inside this process.
            // GLOOP_CUDA_SAFE_CALL(cudaHostUnregister((void*)req.u.munmap.address));
            GLOOP_DEBUG("Done\n");
            {
                std::lock_guard<HostContext::Mutex> guard(m_currentContext->mutex());
                void* host = m_currentContext->table().unregisterMapping((void*)req.u.munmap.address);
                int error = ::munmap(host, req.u.munmap.size);
                ipc->request()->u.munmapResult.error = error;
                ipc->emit(Code::ExitRequired);
                m_currentContext->addExitRequired(ipc);
            }
        });
        break;
    }

    case Code::Msync: {
        // FIXME: Significant naive implementaion.
        // We should integrate implementation with GPUfs's buffer cache.
        m_ioService.post([ipc, req, this]() {
            {
                std::lock_guard<HostContext::Mutex> guard(m_currentContext->mutex());
                void* host = m_currentContext->table().lookupHostByDevice((void*)req.u.msync.address);
                int error = ::msync(host, req.u.msync.size, req.u.msync.flags);
                ipc->request()->u.msyncResult.error = error;
                ipc->emit(Code::Complete);
            }
        });
        break;
    }

    }
    return false;
}

}  // namespace gloop
