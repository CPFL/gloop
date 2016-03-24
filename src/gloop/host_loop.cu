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
#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <boost/thread.hpp>
#include <cassert>
#include <cstdio>
#include <cuda.h>
#include <memory>
#include <sys/mman.h>
#include "benchmark.h"
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
#include "monitor_utility.h"
#include "request.h"
#include "sync_read_write.h"
#include "system_initialize.h"
#include "utility.h"
namespace gloop {

HostLoop::HostLoop(int deviceNumber, uint64_t costPerBit)
    : m_deviceNumber(deviceNumber)
    // , m_loop(uv_loop_new())
    , m_ioService()
    , m_kernelService()
    , m_monitorConnection(m_ioService)
    , m_kernelLock(*this)
{
    // Connect to the gloop monitor.
    {
        m_monitorConnection.connect(boost::asio::local::stream_protocol::endpoint(monitor::createName(GLOOP_ENDPOINT, m_deviceNumber)));
        Command command = {
            .type = Command::Type::Initialize,
            .payload = costPerBit,
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
    m_requestQueue = monitor::createQueue(GLOOP_SHARED_REQUEST_QUEUE, m_id, false);
    m_responseQueue = monitor::createQueue(GLOOP_SHARED_RESPONSE_QUEUE, m_id, false);
    m_sharedMemory = monitor::createMemory(GLOOP_SHARED_MEMORY, m_id, GLOOP_SHARED_MEMORY_SIZE, false);
    m_signal = make_unique<boost::interprocess::mapped_region>(*m_sharedMemory.get(), boost::interprocess::read_write, /* Offset. */ 0, GLOOP_SHARED_MEMORY_SIZE);
    GLOOP_DEBUG("id:(%u)\n", m_id);
}

HostLoop::~HostLoop()
{
    // uv_loop_close(m_loop);
    // GLOOP_DATA_LOG("let's cleanup\n");
    {
        std::lock_guard<KernelLock> lock(m_kernelLock);
        // GLOOP_DATA_LOG("let's cleanup acquire\n");

        stopPoller();

        // Before destroying the primary GPU context,
        // we should clear all the GPU resources.
        m_copyWorkPool.reset();

        CUdevice device;
        GLOOP_CUDA_SAFE_CALL(cuDeviceGet(&device, 0));
        GLOOP_CUDA_SAFE_CALL(cuDevicePrimaryCtxRelease(device));
    }

    {
        boost::unique_lock<boost::mutex> lock(m_threadGroupMutex);
        m_stopThreadGroup = true;
        m_threadGroupNotify.notify_all();
    }
    m_threadGroup.join_all();
    // GLOOP_DATA_LOG("let's cleanup done\n");
}

std::unique_ptr<HostLoop> HostLoop::create(int deviceNumber, uint64_t costPerBit)
{
    gloop::initialize();
    std::unique_ptr<HostLoop> hostLoop(new HostLoop(deviceNumber, costPerBit));
    hostLoop->initialize();
    return hostLoop;
}

// GPU RPC poller.
void HostLoop::runPoller()
{
    assert(!m_poller);
    m_poller = make_unique<boost::thread>([this]() {
        initializeInThread();
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
    uint32_t count = 0;
    while (true) {
        bool found = m_currentContext->tryPeekRequest([&](IPC* ipc) {
            request::Request req { };
            memcpy(&req, (request::Request*)ipc->request(), sizeof(request::Request));
            ipc->emit(Code::None);
            handleIO(ipc, req);
        });
        if (found) {
            count = 0;
            continue;
        }
        // if ((++count % 100000) == 0) {
            boost::this_thread::interruption_point();
        // }
    }
}

bool HostLoop::threadReady()
{
    // The IO threads and the kernel thread.
    boost::unique_lock<boost::mutex> lock(m_threadGroupMutex);
    if (++m_threadGroupReadyCount == (GLOOP_THREAD_GROUP_SIZE + 1)) {
        m_threadGroupReadyNotify.notify_one();
    }
    m_threadGroupNotify.wait(lock);
    if (m_stopThreadGroup) {
        return false;
    }
    return true;
}

void HostLoop::initialize()
{
    initializeInThread();
    {
        // This ensures that primary GPU context is initialized.
        std::lock_guard<KernelLock> lock(m_kernelLock);
        GLOOP_CUDA_SAFE_CALL(cudaSetDeviceFlags(cudaDeviceMapHost | cudaDeviceScheduleSpin));
        GLOOP_CUDA_SAFE_CALL(cudaStreamCreate(&m_pgraph));

        GLOOP_CUDA_SAFE_CALL(cudaHostRegister(m_signal->get_address(), GLOOP_SHARED_MEMORY_SIZE, cudaHostRegisterMapped));
        GLOOP_CUDA_SAFE_CALL(cudaHostGetDevicePointer(&m_deviceSignal, m_signal->get_address(), 0));
        m_copyWorkPool = make_unique<CopyWorkPool>(m_ioService);

        for (int i = 0; i < GLOOP_INITIAL_COPY_WORKS; ++i) {
            m_copyWorkPool->registerCopyWork(CopyWork::create(*this));
        }

        CUDA_SAFE_CALL(cudaPeekAtLastError());

        CUDA_SAFE_CALL(cudaGetDeviceProperties(&m_deviceProperties, m_deviceNumber));
#if 1
        printf("clock rate:(%d)\n", m_deviceProperties.clockRate);
#endif
    }

    // Since kernel work is already held by kernel executing thread,
    // when joining threads, we can say that all the events produced by ASIO
    // is already drained.
    {
        boost::unique_lock<boost::mutex> lock(m_threadGroupMutex);
        for (int i = 0; i < GLOOP_THREAD_GROUP_SIZE; ++i) {
            m_threadGroup.create_thread([this] {
                initializeInThread();
                while (true) {
                    if (!threadReady()) {
                        return;
                    }
                    m_ioService.run();
                }
            });
        }

        m_threadGroup.create_thread([this] {
            initializeInThread();
            while (true) {
                if (!threadReady()) {
                    return;
                }
                m_kernelService.run();
            }
        });

        m_threadGroupReadyNotify.wait(lock);
        m_threadGroupReadyCount = 0;
    }
}

void HostLoop::initializeInThread()
{
    GLOOP_CUDA_SAFE_CALL(cudaSetDevice(m_deviceNumber));
}

void HostLoop::drain()
{
    {
        boost::unique_lock<boost::mutex> lock(m_threadGroupMutex);
        m_threadGroupNotify.notify_all();
        m_threadGroupReadyNotify.wait(lock);
        m_threadGroupReadyCount = 0;
    }
}

void HostLoop::prepareForLaunch()
{
    m_currentContext->prepareForLaunch();
    syncWrite<uint32_t>(static_cast<volatile uint32_t*>(m_signal->get_address()), 0);
}

void HostLoop::resume()
{
    // GLOOP_DEBUG("resume\n");
    m_kernelService.post([&] {
        bool acquireLockSoon = false;
        {
            m_kernelLock.lock();
            // GLOOP_DATA_LOG("acquire for resume\n");
            prepareForLaunch();
            tryLaunch([&] {
                gloop::resume<<<m_currentContext->blocks(), m_threads, 0, m_pgraph>>>(m_deviceSignal, m_currentContext->deviceContext());
            });
            GLOOP_CUDA_SAFE_CALL(cudaStreamSynchronize(m_pgraph));
            acquireLockSoon = m_currentContext->pending();
            m_kernelLock.unlock(acquireLockSoon);
        }
        if (acquireLockSoon) {
            resume();
            return;
        }
        derefKernel();
    });
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
    // logGPUfsDone();
    m_currentContext = nullptr;
}

void HostLoop::refKernel()
{
    m_kernelWork = make_unique<boost::asio::io_service::work>(m_ioService);
}

void HostLoop::derefKernel()
{
    m_kernelWork.reset();
}

CopyWork* HostLoop::acquireCopyWork()
{
    if (CopyWork* work = m_copyWorkPool->tryAcquire()) {
        return work;
    }

    // FIXME: Should acquire lock in the kernel thread.
    // This is a bug.
    std::lock_guard<KernelLock> lock(m_kernelLock);
    std::shared_ptr<CopyWork> work = CopyWork::create(*this);
    m_copyWorkPool->registerCopyWork(work);
    return work.get();
}

void HostLoop::releaseCopyWork(CopyWork* copyWork)
{
    m_copyWorkPool->release(copyWork);
}

bool HostLoop::handleIO(IPC* ipc, request::Request req)
{
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
            CopyWork* copyWork = acquireCopyWork();
            assert(req.u.write.count <= copyWork->hostMemory().size());

            GLOOP_CUDA_SAFE_CALL(cudaMemcpyAsync(copyWork->hostMemory().hostPointer(), req.u.write.buffer, req.u.write.count, cudaMemcpyDeviceToHost, copyWork->stream()));
            GLOOP_CUDA_SAFE_CALL(cudaStreamSynchronize(copyWork->stream()));
            __sync_synchronize();

            ssize_t writtenCount = ::pwrite(req.u.write.fd, copyWork->hostMemory().hostPointer(), req.u.write.count, req.u.write.offset);

            releaseCopyWork(copyWork);

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

            CopyWork* copyWork = acquireCopyWork();
            assert(req.u.read.count <= copyWork->hostMemory().size());
            ssize_t readCount = ::pread(req.u.read.fd, copyWork->hostMemory().hostPointer(), req.u.read.count, req.u.read.offset);

            // FIXME: Should use multiple streams. And execute async.
            GLOOP_CUDA_SAFE_CALL(cudaMemcpyAsync(req.u.read.buffer, copyWork->hostMemory().hostPointer(), readCount, cudaMemcpyHostToDevice, copyWork->stream()));
            GLOOP_CUDA_SAFE_CALL(cudaStreamSynchronize(copyWork->stream()));

            releaseCopyWork(copyWork);

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

    case Code::Ftruncate: {
        m_ioService.post([ipc, req, this]() {
            int result = ::ftruncate(req.u.ftruncate.fd, req.u.ftruncate.offset);
            ipc->request()->u.ftruncateResult.error = result;
            ipc->emit(Code::Complete);
        });
        break;
    }

    case Code::Mmap: {
        // FIXME: Significant naive implementaion.
        // We should integrate implementation with GPUfs's buffer cache.
        m_ioService.post([ipc, req, this]() {
            void* host = ::mmap(req.u.mmap.address, req.u.mmap.size, req.u.mmap.prot, req.u.mmap.flags, req.u.mmap.fd, req.u.mmap.offset);
            // GLOOP_DEBUG("mmap:address:(%p),size:(%u),prot:(%d),flags:(%d),fd:(%d),offset:(%d),res:(%p)\n", req.u.mmap.address, req.u.mmap.size, req.u.mmap.prot, req.u.mmap.flags, req.u.mmap.fd, req.u.mmap.offset, host);
            void* device = nullptr;
            // Not sure, but, mapped memory can be accessed immediately from GPU kernel.
            GLOOP_CUDA_SAFE_CALL(cudaHostRegister(host, req.u.mmap.size, cudaHostRegisterMapped));
            GLOOP_CUDA_SAFE_CALL(cudaHostGetDevicePointer(&device, host, 0));
            {
                std::lock_guard<HostContext::Mutex> guard(m_currentContext->mutex());
                m_currentContext->table().registerMapping(host, device);
                ipc->request()->u.mmapResult.address = device;
                ipc->emit(Code::Complete);
            }
        });
        break;
    }

    case Code::Munmap: {
        // FIXME: Significant naive implementaion.
        // We should integrate implementation with GPUfs's buffer cache.
        m_ioService.post([ipc, req, this]() {
            // GLOOP_DEBUG("munmap:address:(%p),size:(%u)\n", req.u.munmap.address, req.u.munmap.size);
            // FIXME: We should schedule this inside this process.
            {
                std::lock_guard<HostContext::Mutex> guard(m_currentContext->mutex());
                void* host = m_currentContext->table().unregisterMapping((void*)req.u.munmap.address);
                int error = ::munmap(host, req.u.munmap.size);
                m_currentContext->addUnmapRequest((void*)req.u.munmap.address);
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
            // GLOOP_DEBUG("msync:address:(%p),size:(%u),flags:(%u)\n", req.u.msync.address, req.u.msync.size, req.u.msync.flags);
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

    case Code::NetTCPConnect: {
        // GLOOP_DEBUG("net::tcp::connect:address:(%08x),port:(%u)\n", req.u.netTCPConnect.address.sin_addr.s_addr, req.u.netTCPConnect.address.sin_port);
        boost::asio::ip::tcp::socket* socket = new boost::asio::ip::tcp::socket(m_ioService);
        // socket->set_option(boost::asio::socket_base::receive_buffer_size(1 << 20));
        // socket->set_option(boost::asio::socket_base::send_buffer_size(1 << 20));
        socket->async_connect(boost::asio::ip::tcp::endpoint(boost::asio::ip::address_v4(req.u.netTCPConnect.address.sin_addr.s_addr), req.u.netTCPConnect.address.sin_port), [ipc, req, socket, this](const boost::system::error_code& error) {
            if (error) {
                GLOOP_DEBUG("%s\n", error.message().c_str());
                delete socket;
                ipc->request()->u.netTCPConnectResult.socket = nullptr;
            } else {
                ipc->request()->u.netTCPConnectResult.socket = reinterpret_cast<net::Socket*>(socket);
            }
            ipc->emit(Code::Complete);
        });
        break;
    }

    case Code::NetTCPBind: {
        // GLOOP_DEBUG("net::tcp::bind:address:(%08x),port:(%u)\n", req.u.netTCPBind.address.sin_addr.s_addr, req.u.netTCPBind.address.sin_port);
        boost::asio::ip::tcp::acceptor* acceptor = new boost::asio::ip::tcp::acceptor(m_ioService, boost::asio::ip::tcp::endpoint(boost::asio::ip::address_v4(req.u.netTCPBind.address.sin_addr.s_addr), req.u.netTCPBind.address.sin_port));
        ipc->request()->u.netTCPBindResult.server = reinterpret_cast<net::Server*>(acceptor);
        ipc->emit(Code::Complete);
        break;
    }

    case Code::NetTCPUnbind: {
        // GLOOP_DEBUG("net::tcp::unbind:server:(%p)\n", req.u.netTCPUnbind.server);
        assert(reinterpret_cast<boost::asio::ip::tcp::acceptor*>(req.u.netTCPUnbind.server));
        m_ioService.post([ipc, req, this]() {
            delete reinterpret_cast<boost::asio::ip::tcp::acceptor*>(req.u.netTCPUnbind.server);
            ipc->request()->u.netTCPUnbindResult.error = 0;
            ipc->emit(Code::Complete);
        });
        break;
    }

    case Code::NetTCPAccept: {
        // GLOOP_DEBUG("net::tcp::accept:server:(%p)\n", req.u.netTCPAccept.server);
        boost::asio::ip::tcp::socket* socket = new boost::asio::ip::tcp::socket(m_ioService);
        reinterpret_cast<boost::asio::ip::tcp::acceptor*>(req.u.netTCPAccept.server)->async_accept(*socket, [ipc, req, socket, this](const boost::system::error_code& error) {
            if (error) {
                delete socket;
                ipc->request()->u.netTCPAcceptResult.socket = nullptr;
            } else {
                ipc->request()->u.netTCPAcceptResult.socket = reinterpret_cast<net::Socket*>(socket);
            }
            ipc->emit(Code::Complete);
        });
        break;
    }

    case Code::NetTCPReceive: {
//         std::shared_ptr<gloop::Benchmark> benchmark = std::make_shared<gloop::Benchmark>();
//         benchmark->begin();
        // GLOOP_DEBUG("net::tcp::receive:socket:(%p),count:(%u),buffer:(%p)\n", req.u.netTCPReceive.socket, req.u.netTCPReceive.count, req.u.netTCPReceive.buffer);
        assert(reinterpret_cast<boost::asio::ip::tcp::socket*>(req.u.netTCPReceive.socket));
        // FIXME: This should be reconsidered.
        size_t count = req.u.netTCPReceive.count;
        assert(count <= GLOOP_SHARED_PAGE_SIZE);
        boost::asio::ip::tcp::socket* socket = reinterpret_cast<boost::asio::ip::tcp::socket*>(req.u.netTCPReceive.socket);
        CopyWork* copyWork = acquireCopyWork();
        boost::asio::async_read(*socket, boost::asio::buffer(copyWork->hostMemory().hostPointer(), count), boost::asio::transfer_at_least(1), [=](const boost::system::error_code& error, size_t receiveCount) {
            if (error) {
                if ((boost::asio::error::eof == error) || (boost::asio::error::connection_reset == error)) {
                    ipc->request()->u.netTCPReceiveResult.receiveCount = 0;
                } else {
                    ipc->request()->u.netTCPReceiveResult.receiveCount = -1;
                }
            } else {
                GLOOP_CUDA_SAFE_CALL(cudaMemcpyAsync(req.u.netTCPReceive.buffer, copyWork->hostMemory().hostPointer(), receiveCount, cudaMemcpyHostToDevice, copyWork->stream()));
                ipc->request()->u.netTCPReceiveResult.receiveCount = receiveCount;
                GLOOP_CUDA_SAFE_CALL(cudaStreamSynchronize(copyWork->stream()));
            }
            releaseCopyWork(copyWork);
            ipc->emit(Code::Complete);
//             benchmark->end();
//             std::printf("receive: count:(%u),ticks:(%u)\n", count, benchmark->ticks().count());
        });
        break;
    }

    case Code::NetTCPSend: {
        // GLOOP_DEBUG("net::tcp::send:socket:(%p),count:(%u),buffer:(%p)\n", req.u.netTCPSend.socket, req.u.netTCPSend.count, req.u.netTCPSend.buffer);
        assert(reinterpret_cast<boost::asio::ip::tcp::socket*>(req.u.netTCPSend.socket));
        m_ioService.post([=]() {
            // FIXME: This should be reconsidered.
            size_t count = req.u.netTCPSend.count;

            CopyWork* copyWork = acquireCopyWork();
            assert(count <= copyWork->hostMemory().size());

//             std::shared_ptr<gloop::Benchmark> benchmark = std::make_shared<gloop::Benchmark>();
//             benchmark->begin();
            GLOOP_CUDA_SAFE_CALL(cudaMemcpyAsync(copyWork->hostMemory().hostPointer(), req.u.netTCPSend.buffer, count, cudaMemcpyDeviceToHost, copyWork->stream()));
            GLOOP_CUDA_SAFE_CALL(cudaStreamSynchronize(copyWork->stream()));
            boost::asio::ip::tcp::socket* socket = reinterpret_cast<boost::asio::ip::tcp::socket*>(req.u.netTCPSend.socket);
            boost::asio::async_write(*socket, boost::asio::buffer(copyWork->hostMemory().hostPointer(), count), [=](const boost::system::error_code& error, size_t sentCount) {
                releaseCopyWork(copyWork);
                if (error) {
                    if ((boost::asio::error::eof == error) || (boost::asio::error::connection_reset == error)) {
                        ipc->request()->u.netTCPReceiveResult.receiveCount = 0;
                    } else {
                        ipc->request()->u.netTCPReceiveResult.receiveCount = -1;
                    }
                } else {
                    ipc->request()->u.netTCPSendResult.sentCount = sentCount;
                }
                ipc->emit(Code::Complete);
//                 benchmark->end();
//                 std::printf("send: count:(%u),ticks:(%u)\n", count, benchmark->ticks().count());
            });
        });
        break;
    }

    case Code::NetTCPClose: {
        // GLOOP_DEBUG("net::tcp::close:socket:(%p)\n", req.u.netTCPClose.socket);
        assert(reinterpret_cast<boost::asio::ip::tcp::socket*>(req.u.netTCPClose.socket));
        m_ioService.post([ipc, req, this]() {
            delete reinterpret_cast<boost::asio::ip::tcp::socket*>(req.u.netTCPClose.socket);
            ipc->request()->u.netTCPCloseResult.error = 0;
            ipc->emit(Code::Complete);
        });
        break;
    }

    case Code::Exit: {
        ipc->emit(Code::ExitRequired);
        m_currentContext->addExitRequired(ipc);
        break;
    }

    }
    return false;
}

}  // namespace gloop
