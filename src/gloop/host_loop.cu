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
#include "benchmark.h"
#include "bitwise_cast.h"
#include "command.h"
#include "config.h"
#include "copy_work_pool.cuh"
#include "copy_worker.cuh"
#include "data_log.h"
#include "helper.cuh"
#include "host_loop_inlines.cuh"
#include "initialize.cuh"
#include "io.cuh"
#include "make_unique.h"
#include "memcpy_io.cuh"
#include "monitor_utility.h"
#include "request.h"
#include "rpc.cuh"
#include "sync_read_write.h"
#include "system_initialize.h"
#include "utility.h"
#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <boost/thread.hpp>
#include <cassert>
#include <cstdio>
#include <cuda.h>
#include <memory>
#include <sys/mman.h>
namespace gloop {

GLOOP_ALWAYS_INLINE static void emit(const std::lock_guard<HostContext::Mutex>&, HostContext& context, RPC rpc, Code code)
{
    rpc.emit(context, code);
    context.condition().notify_one();
}

GLOOP_ALWAYS_INLINE static void emit(HostContext& context, RPC rpc, Code code)
{
    std::lock_guard<HostContext::Mutex> lock(context.mutex());
    emit(lock, context, rpc, code);
}

HostLoop::HostLoop(int deviceNumber, uint64_t costPerBit)
    : m_deviceNumber(deviceNumber)
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
        Command result{};
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
    // GLOOP_DATA_LOG("let's cleanup\n");
    {
        std::lock_guard<KernelLock> lock(m_kernelLock);
        // GLOOP_DATA_LOG("let's cleanup acquire\n");

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

#if 0
__global__ void initializeHostLoop()
{
}
#endif

void HostLoop::initialize()
{
    {
        // This ensures that primary GPU context is initialized.
        std::lock_guard<KernelLock> lock(m_kernelLock);
        GLOOP_CUDA_SAFE_CALL(cudaSetDevice(m_deviceNumber));
        // GLOOP_CUDA_SAFE_CALL(cudaSetDeviceFlags(cudaDeviceMapHost | cudaDeviceScheduleSpin));
        GLOOP_CUDA_SAFE_CALL(cudaSetDeviceFlags(cudaDeviceMapHost));
        eagerlyInitializeContext();

        GLOOP_CUDA_SAFE_CALL(cudaStreamCreate(&m_pgraph));

        GLOOP_CUDA_SAFE_CALL(cudaHostRegister(m_signal->get_address(), GLOOP_SHARED_MEMORY_SIZE, cudaHostRegisterMapped));
        GLOOP_CUDA_SAFE_CALL(cudaHostGetDevicePointer(&m_deviceSignal, m_signal->get_address(), 0));
        m_copyWorkPool = make_unique<CopyWorkPool>(m_ioService);

        for (int i = 0; i < GLOOP_INITIAL_COPY_WORKS; ++i) {
            m_copyWorkPool->registerCopyWork(CopyWork::create(*this));
        }

        GLOOP_CUDA_SAFE_CALL(cudaPeekAtLastError());

        GLOOP_CUDA_SAFE_CALL(cudaGetDeviceProperties(&m_deviceProperties, m_deviceNumber));

        // And initialize launch related things eagerly here.
        // initializeHostLoop<<<1, 1, 0, m_pgraph>>>();
        // cudaStreamSynchronize(m_pgraph);
        GLOOP_CUDA_SAFE_CALL(cudaMemcpyToSymbol(gloop::signal, &m_deviceSignal, sizeof(m_deviceSignal)));
        // GLOOP_DATA_LOG("clock rate:(%d)\n", m_deviceProperties.clockRate);
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
                    m_ioService.reset();
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
                m_kernelService.reset();
                m_kernelService.run();
            }
        });

        m_threadGroupReadyNotify.wait(lock);
        m_threadGroupReadyCount = 0;
    }
}

void HostLoop::initializeInThread()
{
    // std::lock_guard<KernelLock> lock(m_kernelLock);
    GLOOP_CUDA_SAFE_CALL(cudaSetDevice(m_deviceNumber));
    // eagerlyInitializeContext();
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

void HostLoop::prepareForLaunch(HostContext& hostContext)
{
    hostContext.prepareForLaunch();
    syncWrite<uint32_t>(static_cast<volatile uint32_t*>(m_signal->get_address()), 0);
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

bool HostLoop::handleIO(HostContext& context, RPC rpc, Code code, request::Request req)
{
    switch (static_cast<Code>(code)) {
    case Code::Open: {
        int fd = 0;
        {
            std::lock_guard<HostContext::Mutex> guard(context.mutex());
            fd = context.table().open(req.u.open.filename.data, req.u.open.mode);
        }
        // GLOOP_DATA_LOG("open:(%s),fd:(%d)\n", req.u.open.filename.data, fd);
        rpc.request(context)->u.openResult.fd = fd;
        emit(context, rpc, Code::Complete);
        break;
    }

    case Code::Write: {
        // FIXME: Significant naive implementaion.
        // We should integrate implementation with GPUfs's buffer cache.
        m_ioService.post([rpc, req, this, &context] {
            // GLOOP_DEBUG("Write fd:(%d),count:(%u),offset:(%d),page:(%p)\n", req.u.write.fd, (unsigned)req.u.write.count, (int)req.u.write.offset, (void*)req.u.read.buffer);
            CopyWork* copyWork = acquireCopyWork();
            assert(req.u.write.count <= copyWork->hostMemory().size());

            GLOOP_CUDA_SAFE_CALL(cudaMemcpyAsync(copyWork->hostMemory().hostPointer(), req.u.write.buffer, req.u.write.count, cudaMemcpyDeviceToHost, copyWork->stream()));
            GLOOP_CUDA_SAFE_CALL(cudaStreamSynchronize(copyWork->stream()));
            __sync_synchronize();

            ssize_t writtenCount = ::pwrite(req.u.write.fd, copyWork->hostMemory().hostPointer(), req.u.write.count, req.u.write.offset);

            releaseCopyWork(copyWork);

            rpc.request(context)->u.writeResult.writtenCount = writtenCount;
            emit(context, rpc, Code::Complete);
        });
        break;
    }

    case Code::Read: {
        // FIXME: Significant naive implementaion.
        // We should integrate implementation with GPUfs's buffer cache.
        m_ioService.post([rpc, req, this, &context] {
            // GLOOP_DATA_LOG("Read rpc:(%p),fd:(%d),count:(%u),offset(%d),page:(%p)\n", (void*)rpc, req.u.read.fd, (unsigned)req.u.read.count, (int)req.u.read.offset, (void*)req.u.read.buffer);

            CopyWork* copyWork = acquireCopyWork();
            assert(req.u.read.count <= copyWork->hostMemory().size());
            ssize_t readCount = ::pread(req.u.read.fd, copyWork->hostMemory().hostPointer(), req.u.read.count, req.u.read.offset);

            // FIXME: Should use multiple streams. And execute async.
            assert(req.u.read.buffer);
            GLOOP_CUDA_SAFE_CALL(cudaMemcpyAsync(req.u.read.buffer, copyWork->hostMemory().hostPointer(), readCount, cudaMemcpyHostToDevice, copyWork->stream()));
            GLOOP_CUDA_SAFE_CALL(cudaStreamSynchronize(copyWork->stream()));

            releaseCopyWork(copyWork);

            rpc.request(context)->u.readResult.readCount = readCount;
            emit(context, rpc, Code::Complete);
        });
        break;
    }

    case Code::Fstat: {
        struct stat buf {
        };
        ::fstat(req.u.fstat.fd, &buf);
        // GLOOP_DEBUG("Fstat %d %u\n", req.u.fstat.fd, buf.st_size);
        rpc.request(context)->u.fstatResult.size = buf.st_size;
        emit(context, rpc, Code::Complete);
        break;
    }

    case Code::Close: {
        {
            std::lock_guard<HostContext::Mutex> guard(context.mutex());
            context.table().close(req.u.close.fd);
        }
        // GLOOP_DATA_LOG("Close %d\n", req.u.close.fd);
        rpc.request(context)->u.closeResult.error = 0;
        emit(context, rpc, Code::Complete);
        break;
    }

    case Code::Ftruncate: {
        m_ioService.post([rpc, req, this, &context] {
            int result = ::ftruncate(req.u.ftruncate.fd, req.u.ftruncate.offset);
            rpc.request(context)->u.ftruncateResult.error = result;
            emit(context, rpc, Code::Complete);
        });
        break;
    }

    case Code::Mmap: {
        // FIXME: Significant naive implementaion.
        // We should integrate implementation with GPUfs's buffer cache.
        m_ioService.post([rpc, req, this, &context] {
            void* device = nullptr;
            {
                std::lock_guard<HostContext::Mutex> guard(context.mutex());
                std::shared_ptr<MmapResult> result;
                if (context.table().requestMmap(req.u.mmap.fd, req.u.mmap.offset, req.u.mmap.size, result)) {
                    // New!
                    size_t size = req.u.mmap.size;
                    void* host = ::mmap(req.u.mmap.address, req.u.mmap.size, req.u.mmap.prot, req.u.mmap.flags, req.u.mmap.fd, req.u.mmap.offset);
                    // GLOOP_DATA_LOG("mmap:address:(%p),size:(%u),prot:(%d),flags:(%d),fd:(%d),offset:(%d),res:(%p)\n", req.u.mmap.address, req.u.mmap.size, req.u.mmap.prot, req.u.mmap.flags, req.u.mmap.fd, req.u.mmap.offset, host);
                    void* device = nullptr;
                    // Not sure, but, mapped memory can be accessed immediately from GPU kernel.
                    GLOOP_CUDA_SAFE_CALL(cudaHostRegister(host, size, cudaHostRegisterMapped));
                    GLOOP_CUDA_SAFE_CALL(cudaHostGetDevicePointer(&device, host, 0));
                    result->host = host;
                    result->device = device;
                    result->size = size;
                    context.table().registerMapping(device, result);
                }
                device = result->device;
                rpc.request(context)->u.mmapResult.address = device;
                emit(guard, context, rpc, Code::Complete);
            }
            // GLOOP_DATA_LOG("mmap:device(%p)\n", device);
        });
        break;
    }

    case Code::Munmap: {
        // FIXME: Significant naive implementaion.
        // We should integrate implementation with GPUfs's buffer cache.
        m_ioService.post([rpc, req, this, &context] {
            // GLOOP_DATA_LOG("munmap:address:(%p),size:(%u)\n", req.u.munmap.address, req.u.munmap.size);
            // FIXME: We should schedule this inside this process.
            {
                std::lock_guard<HostContext::Mutex> guard(context.mutex());
                std::shared_ptr<MmapResult> result = context.table().unregisterMapping((void*)req.u.munmap.address);
                Code code = Code::Complete;
                int error = 0;
                if (result) {
                    if (!result->refCount) {
                        // error = ::munmap(result.host, req.u.munmap.size);
                        context.addUnmapRequest(guard, result);
                        code = Code::ExitRequired;
                        context.addExitRequired(guard, rpc);
                    }
                } else {
                    error = -EINVAL;
                }
                rpc.request(context)->u.munmapResult.error = error;
                emit(guard, context, rpc, code);
                if (code == Code::ExitRequired) {
                    syncWrite<uint32_t>(static_cast<volatile uint32_t*>(m_signal->get_address()), 1);
                }
            }
        });
        break;
    }

    case Code::Msync: {
        // FIXME: Significant naive implementaion.
        // We should integrate implementation with GPUfs's buffer cache.
        m_ioService.post([rpc, req, this, &context] {
            // GLOOP_DEBUG("msync:address:(%p),size:(%u),flags:(%u)\n", req.u.msync.address, req.u.msync.size, req.u.msync.flags);
            void* host = nullptr;
            {
                std::lock_guard<HostContext::Mutex> guard(context.mutex());
                host = context.table().lookupHostByDevice((void*)req.u.msync.address);
            }
            int error = ::msync(host, req.u.msync.size, req.u.msync.flags);
            rpc.request(context)->u.msyncResult.error = error;
            emit(context, rpc, Code::Complete);
        });
        break;
    }

    case Code::NetTCPConnect: {
        // GLOOP_DEBUG("net::tcp::connect:address:(%08x),port:(%u)\n", req.u.netTCPConnect.address.sin_addr.s_addr, req.u.netTCPConnect.address.sin_port);
        boost::asio::ip::tcp::socket* socket = new boost::asio::ip::tcp::socket(m_ioService);
        // socket->set_option(boost::asio::socket_base::receive_buffer_size(1 << 20));
        // socket->set_option(boost::asio::socket_base::send_buffer_size(1 << 20));
        socket->async_connect(boost::asio::ip::tcp::endpoint(boost::asio::ip::address_v4(req.u.netTCPConnect.address.sin_addr.s_addr), req.u.netTCPConnect.address.sin_port), [rpc, req, socket, this, &context](const boost::system::error_code& error) {
            if (error) {
                GLOOP_DEBUG("%s\n", error.message().c_str());
                delete socket;
                rpc.request(context)->u.netTCPConnectResult.socket = nullptr;
            } else {
                rpc.request(context)->u.netTCPConnectResult.socket = reinterpret_cast<net::Socket*>(socket);
            }
            emit(context, rpc, Code::Complete);
        });
        break;
    }

    case Code::NetTCPBind: {
        // GLOOP_DEBUG("net::tcp::bind:address:(%08x),port:(%u)\n", req.u.netTCPBind.address.sin_addr.s_addr, req.u.netTCPBind.address.sin_port);
        boost::asio::ip::tcp::acceptor* acceptor = new boost::asio::ip::tcp::acceptor(m_ioService, boost::asio::ip::tcp::endpoint(boost::asio::ip::address_v4(req.u.netTCPBind.address.sin_addr.s_addr), req.u.netTCPBind.address.sin_port));
        rpc.request(context)->u.netTCPBindResult.server = reinterpret_cast<net::Server*>(acceptor);
        emit(context, rpc, Code::Complete);
        break;
    }

    case Code::NetTCPUnbind: {
        // GLOOP_DEBUG("net::tcp::unbind:server:(%p)\n", req.u.netTCPUnbind.server);
        assert(reinterpret_cast<boost::asio::ip::tcp::acceptor*>(req.u.netTCPUnbind.server));
        m_ioService.post([rpc, req, this, &context] {
            delete reinterpret_cast<boost::asio::ip::tcp::acceptor*>(req.u.netTCPUnbind.server);
            rpc.request(context)->u.netTCPUnbindResult.error = 0;
            emit(context, rpc, Code::Complete);
        });
        break;
    }

    case Code::NetTCPAccept: {
        // GLOOP_DATA_LOG("net::tcp::accept:server:(%p)\n", req.u.netTCPAccept.server);
        boost::asio::ip::tcp::socket* socket = new boost::asio::ip::tcp::socket(m_ioService);
        reinterpret_cast<boost::asio::ip::tcp::acceptor*>(req.u.netTCPAccept.server)->async_accept(*socket, [rpc, req, socket, this, &context](const boost::system::error_code& error) {
            if (error) {
                delete socket;
                rpc.request(context)->u.netTCPAcceptResult.socket = nullptr;
            } else {
                rpc.request(context)->u.netTCPAcceptResult.socket = reinterpret_cast<net::Socket*>(socket);
            }
            emit(context, rpc, Code::Complete);
        });
        break;
    }

    case Code::NetTCPReceive: {
        //         std::shared_ptr<gloop::Benchmark> benchmark = std::make_shared<gloop::Benchmark>();
        //         benchmark->begin();
        // GLOOP_DATA_LOG("net::tcp::receive:socket:(%p),count:(%u),buffer:(%p)\n", req.u.netTCPReceive.socket, req.u.netTCPReceive.count, req.u.netTCPReceive.buffer);
        assert(reinterpret_cast<boost::asio::ip::tcp::socket*>(req.u.netTCPReceive.socket));
        // FIXME: This should be reconsidered.
        size_t count = req.u.netTCPReceive.count;
        assert(count <= GLOOP_SHARED_PAGE_SIZE);
        boost::asio::ip::tcp::socket* socket = reinterpret_cast<boost::asio::ip::tcp::socket*>(req.u.netTCPReceive.socket);
        CopyWork* copyWork = acquireCopyWork();
        auto callback = [=, &context](const boost::system::error_code& error, size_t receiveCount) {
            if (error) {
                if ((boost::asio::error::eof == error) || (boost::asio::error::connection_reset == error)) {
                    rpc.request(context)->u.netTCPReceiveResult.receiveCount = 0;
                } else {
                    rpc.request(context)->u.netTCPReceiveResult.receiveCount = -1;
                }
            } else {
                GLOOP_CUDA_SAFE_CALL(cudaMemcpyAsync(req.u.netTCPReceive.buffer, copyWork->hostMemory().hostPointer(), receiveCount, cudaMemcpyHostToDevice, copyWork->stream()));
                rpc.request(context)->u.netTCPReceiveResult.receiveCount = receiveCount;
                GLOOP_CUDA_SAFE_CALL(cudaStreamSynchronize(copyWork->stream()));
            }
            releaseCopyWork(copyWork);
            emit(context, rpc, Code::Complete);
            //             benchmark->end();
            //             GLOOP_DATA_LOG("receive: count:(%u),ticks:(%u)\n", count, benchmark->ticks().count());
        };
        if (req.u.netTCPReceive.flags & MSG_WAITALL) {
            boost::asio::async_read(*socket, boost::asio::buffer(copyWork->hostMemory().hostPointer(), count), boost::asio::transfer_all(), callback);
        } else {
            boost::asio::async_read(*socket, boost::asio::buffer(copyWork->hostMemory().hostPointer(), count), boost::asio::transfer_at_least(1), callback);
        }
        break;
    }

    case Code::NetTCPSend: {
        // GLOOP_DATA_LOG("net::tcp::send:socket:(%p),count:(%u),buffer:(%p)\n", req.u.netTCPSend.socket, req.u.netTCPSend.count, req.u.netTCPSend.buffer);
        assert(reinterpret_cast<boost::asio::ip::tcp::socket*>(req.u.netTCPSend.socket));
        m_ioService.post([=, &context] {
            // FIXME: This should be reconsidered.
            size_t count = req.u.netTCPSend.count;

            CopyWork* copyWork = acquireCopyWork();
            assert(count <= copyWork->hostMemory().size());

            //             std::shared_ptr<gloop::Benchmark> benchmark = std::make_shared<gloop::Benchmark>();
            //             benchmark->begin();
            GLOOP_CUDA_SAFE_CALL(cudaMemcpyAsync(copyWork->hostMemory().hostPointer(), req.u.netTCPSend.buffer, count, cudaMemcpyDeviceToHost, copyWork->stream()));
            GLOOP_CUDA_SAFE_CALL(cudaStreamSynchronize(copyWork->stream()));
            boost::asio::ip::tcp::socket* socket = reinterpret_cast<boost::asio::ip::tcp::socket*>(req.u.netTCPSend.socket);
            boost::asio::async_write(*socket, boost::asio::buffer(copyWork->hostMemory().hostPointer(), count), [=, &context](const boost::system::error_code& error, size_t sentCount) {
                releaseCopyWork(copyWork);
                if (error) {
                    if ((boost::asio::error::eof == error) || (boost::asio::error::connection_reset == error)) {
                        rpc.request(context)->u.netTCPReceiveResult.receiveCount = 0;
                    } else {
                        rpc.request(context)->u.netTCPReceiveResult.receiveCount = -1;
                    }
                } else {
                    rpc.request(context)->u.netTCPSendResult.sentCount = sentCount;
                }
                emit(context, rpc, Code::Complete);
                //                 benchmark->end();
                //                 GLOOP_DATA_LOG("send: count:(%u),ticks:(%u)\n", count, benchmark->ticks().count());
            });
        });
        break;
    }

    case Code::NetTCPClose: {
        // GLOOP_DEBUG("net::tcp::close:socket:(%p)\n", req.u.netTCPClose.socket);
        assert(reinterpret_cast<boost::asio::ip::tcp::socket*>(req.u.netTCPClose.socket));
        m_ioService.post([rpc, req, this, &context] {
            delete reinterpret_cast<boost::asio::ip::tcp::socket*>(req.u.netTCPClose.socket);
            rpc.request(context)->u.netTCPCloseResult.error = 0;
            emit(context, rpc, Code::Complete);
        });
        break;
    }

    case Code::Exit: {
        std::lock_guard<HostContext::Mutex> guard(context.mutex());
        emit(guard, context, rpc, Code::ExitRequired);
        context.addExitRequired(guard, rpc);
        break;
    }
    }
    return false;
}

} // namespace gloop
