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
#ifndef GLOOP_FS_H_
#define GLOOP_FS_H_
#include <sys/mman.h>
#include <type_traits>
#include <utility>
#include "device_loop.cuh"
#include "memcpy_io.cuh"
#include "request.h"
#include "utility/util.cu.h"

namespace gloop {
namespace fs {

template<typename Lambda>
inline __device__ auto open(DeviceLoop* loop, const char* filename, int mode, Lambda callback) -> void
{
    BEGIN_SINGLE_THREAD
    {
        auto ipc = loop->enqueueIPC([callback](DeviceLoop* loop, volatile request::Request* req) {
            callback(loop, req->u.openResult.fd);
        });
        volatile request::Open& req = ipc.request(loop)->u.open;
        memcpyIO(req.filename.data, filename, GLOOP_FILENAME_SIZE);
        req.mode = mode;
        ipc.emit(loop, Code::Open);
    }
    END_SINGLE_THREAD
}

template<typename Lambda>
inline __device__ auto fstat(DeviceLoop* loop, int fd, Lambda callback) -> void
{
    BEGIN_SINGLE_THREAD
    {
        auto ipc = loop->enqueueIPC([callback](DeviceLoop* loop, volatile request::Request* req) {
            callback(loop, req->u.fstatResult.size);
        });
        volatile request::Fstat& req = ipc.request(loop)->u.fstat;
        req.fd = fd;
        ipc.emit(loop, Code::Fstat);
    }
    END_SINGLE_THREAD
}

template<typename Lambda>
inline __device__ auto close(DeviceLoop* loop, int fd, Lambda callback) -> void
{
    BEGIN_SINGLE_THREAD
    {
        auto ipc = loop->enqueueIPC([callback](DeviceLoop* loop, volatile request::Request* req) {
            callback(loop, req->u.closeResult.error);
        });
        volatile request::Close& req = ipc.request(loop)->u.close;
        req.fd = fd;
        ipc.emit(loop, Code::Close);
    }
    END_SINGLE_THREAD
}

template<typename Lambda>
inline __device__ auto ftruncate(DeviceLoop* loop, int fd, off_t offset, Lambda callback) -> void
{
    BEGIN_SINGLE_THREAD
    {
        auto ipc = loop->enqueueIPC([callback](DeviceLoop* loop, volatile request::Request* req) {
            callback(loop, req->u.ftruncateResult.error);
        });
        volatile request::Ftruncate& req = ipc.request(loop)->u.ftruncate;
        req.fd = fd;
        req.offset = offset;
        ipc.emit(loop, Code::Ftruncate);
    }
    END_SINGLE_THREAD
}

template<typename Lambda>
inline __device__ auto readOnePage(DeviceLoop* loop, int fd, size_t offset, size_t count, Lambda callback) -> void
{
    loop->allocOnePage([=](DeviceLoop* loop, void* page) {
        BEGIN_SINGLE_THREAD
        {
            auto ipc = loop->enqueueIPC([=](DeviceLoop* loop, volatile request::Request* req) {
                __threadfence();
                callback(loop, req->u.readResult.readCount, page);
            });
            volatile request::Read& req = ipc.request(loop)->u.read;
            req.fd = fd;
            req.offset = offset;
            req.count = count;
            req.buffer = static_cast<unsigned char*>(page);
            ipc.emit(loop, Code::Read);
        }
        END_SINGLE_THREAD
    });
}

template<typename Lambda>
inline __device__ auto performOnePageRead(DeviceLoop* loop, int fd, size_t offset, size_t count, unsigned char* buffer, size_t requestedOffset, ssize_t readCount, void* page, Lambda callback) -> void
{
    ssize_t cursor = requestedOffset + readCount;
    ssize_t last = offset + count;

    GPU_ASSERT(readCount <= count);
    GPU_ASSERT(cursor <= last);
    if (readCount < 0) {
        callback(loop, -1);
        return;
    }

    if (cursor != last) {
        readOnePage(loop, fd, cursor, min((last - cursor), GLOOP_SHARED_PAGE_SIZE), [=](DeviceLoop* loop, ssize_t readCount, void* page) {
            performOnePageRead(loop, fd, offset, count, buffer, cursor, readCount, page, callback);
        });
    }

    gpunet::copy_block_src_volatile(buffer + (requestedOffset - offset), reinterpret_cast<volatile uchar*>(page), readCount);
    BEGIN_SINGLE_THREAD
    {
        loop->freeOnePage(page);
    }
    END_SINGLE_THREAD

    if (cursor == last) {
        // Ensure buffer's modification is flushed.
        __threadfence_system();
        callback(loop, count);
    }
}

template<typename Lambda>
inline __device__ auto read(DeviceLoop* loop, int fd, size_t offset, size_t count, unsigned char* buffer, Lambda callback) -> void
{
    readOnePage(loop, fd, offset, min(count, GLOOP_SHARED_PAGE_SIZE), [=](DeviceLoop* loop, ssize_t readCount, void* page) {
        performOnePageRead(loop, fd, offset, count, buffer, offset, readCount, page, callback);
    });
}

template<typename Lambda>
inline __device__ auto writeOnePage(DeviceLoop* loop, int fd, size_t offset, size_t transferringSize, unsigned char* buffer, Lambda callback) -> void
{
    loop->allocOnePage([=](DeviceLoop* loop, void* page) {
        gpunet::copy_block_dst_volatile(reinterpret_cast<volatile uchar*>(page), buffer, transferringSize);
        BEGIN_SINGLE_THREAD
        {
            auto ipc = loop->enqueueIPC([=](DeviceLoop* loop, volatile request::Request* req) {
                BEGIN_SINGLE_THREAD
                {
                    loop->freeOnePage(page);
                }
                END_SINGLE_THREAD
                callback(loop, req->u.writeResult.writtenCount);
            });
            volatile request::Write& req = ipc.request(loop)->u.write;
            req.fd = fd;
            req.offset = offset;
            req.count = transferringSize;
            req.buffer = static_cast<unsigned char*>(page);
            ipc.emit(loop, Code::Write);
        }
        END_SINGLE_THREAD
    });
}

template<typename Lambda>
inline __device__ auto performOnePageWrite(DeviceLoop* loop, int fd, size_t offset, size_t count, unsigned char* buffer, size_t requestedOffset, ssize_t writtenCount, Lambda callback) -> void
{
    ssize_t cursor = requestedOffset + writtenCount;
    ssize_t last = offset + count;

    GPU_ASSERT(writtenCount <= count);
    GPU_ASSERT(cursor <= last);
    if (writtenCount < 0) {
        callback(loop, -1);
        return;
    }

    if (cursor != last) {
        writeOnePage(loop, fd, cursor, min((last - cursor), GLOOP_SHARED_PAGE_SIZE), buffer + (cursor - offset), [=](DeviceLoop* loop, ssize_t writtenCount) {
            performOnePageWrite(loop, fd, offset, count, buffer, cursor, writtenCount, callback);
        });
    } else {
        callback(loop, count);
    }
}


template<typename Lambda>
inline __device__ auto write(DeviceLoop* loop, int fd, size_t offset, size_t count, unsigned char* buffer, Lambda callback) -> void
{
    // Ensure buffer's modification is flushed.
    __threadfence_system();
    writeOnePage(loop, fd, offset, min(count, GLOOP_SHARED_PAGE_SIZE), buffer, [=](DeviceLoop* loop, ssize_t writtenCount) {
        performOnePageWrite(loop, fd, offset, count, buffer, offset, writtenCount, callback);
    });
}

template<typename Lambda>
inline __device__ auto mmap(DeviceLoop* loop, void* address, size_t size, int prot, int flags, int fd, off_t offset, Lambda callback) -> void
{
    BEGIN_SINGLE_THREAD
    {
        auto ipc = loop->enqueueIPC([callback](DeviceLoop* loop, volatile request::Request* req) {
            callback(loop, req->u.mmapResult.address);
        });
        volatile request::Mmap& req = ipc.request(loop)->u.mmap;
        req.address = address;
        req.size = size;
        req.prot = prot;
        req.flags = flags;
        req.fd = fd;
        req.offset = offset;
        ipc.emit(loop, Code::Mmap);
    }
    END_SINGLE_THREAD
}


template<typename Lambda>
inline __device__ auto munmap(DeviceLoop* loop, volatile void* address, size_t size, Lambda callback) -> void
{
    BEGIN_SINGLE_THREAD
    {
        auto ipc = loop->enqueueIPC([callback](DeviceLoop* loop, volatile request::Request* req) {
            callback(loop, req->u.munmapResult.error);
        });
        volatile request::Munmap& req = ipc.request(loop)->u.munmap;
        req.address = address;
        req.size = size;
        ipc.emit(loop, Code::Munmap);
    }
    END_SINGLE_THREAD
}

template<typename Lambda>
inline __device__ auto msync(DeviceLoop* loop, volatile void* address, size_t size, int flags, Lambda callback) -> void
{
    BEGIN_SINGLE_THREAD
    {
        auto ipc = loop->enqueueIPC([callback](DeviceLoop* loop, volatile request::Request* req) {
            callback(loop, req->u.msyncResult.error);
        });
        volatile request::Msync& req = ipc.request(loop)->u.msync;
        req.address = address;
        req.size = size;
        req.flags = flags;
        ipc.emit(loop, Code::Msync);
    }
    END_SINGLE_THREAD
}

} }  // namespace gloop::fs
#endif  // GLOOP_FS_H_
