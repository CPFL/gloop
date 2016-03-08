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

__device__ void openImpl(DeviceLoop*, IPC*, volatile request::Open&, const char* filename, int mode);
__device__ void writeImpl(DeviceLoop*, IPC*, volatile request::Write&, int fd, size_t offset, size_t count, unsigned char* buffer);
__device__ void fstatImpl(DeviceLoop*, IPC*, volatile request::Fstat&, int fd);
__device__ void closeImpl(DeviceLoop*, IPC*, volatile request::Close&, int fd);
__device__ void readImpl(DeviceLoop*, IPC*, volatile request::Read&, int fd, size_t offset, size_t count, unsigned char* buffer);
__device__ void ftruncateImpl(DeviceLoop*, IPC*, volatile request::Ftruncate&, int fd, off_t offset);
__device__ void mmapImpl(DeviceLoop*, IPC*, volatile request::Mmap&, void* address, size_t size, int prot, int flags, int fd, off_t offset);
__device__ void munmapImpl(DeviceLoop*, IPC*, volatile request::Munmap&, volatile void* address, size_t size);
__device__ void msyncImpl(DeviceLoop*, IPC*, volatile request::Msync&, volatile void* address, size_t size, int flags);

template<typename Lambda>
inline __device__ auto open(DeviceLoop* loop, const char* filename, int mode, Lambda callback) -> void
{
    BEGIN_SINGLE_THREAD
    {
        auto* ipc = loop->enqueueIPC([callback](DeviceLoop* loop, volatile request::Request* req) {
            callback(loop, req->u.openResult.fd);
        });
        openImpl(loop, ipc, ipc->request()->u.open, filename, mode);
    }
    END_SINGLE_THREAD
}

template<typename Lambda>
inline __device__ auto fstat(DeviceLoop* loop, int fd, Lambda callback) -> void
{
    BEGIN_SINGLE_THREAD
    {
        auto* ipc = loop->enqueueIPC([callback](DeviceLoop* loop, volatile request::Request* req) {
            callback(loop, req->u.fstatResult.size);
        });
        fstatImpl(loop, ipc, ipc->request()->u.fstat, fd);
    }
    END_SINGLE_THREAD
}

template<typename Lambda>
inline __device__ auto close(DeviceLoop* loop, int fd, Lambda callback) -> void
{
    BEGIN_SINGLE_THREAD
    {
        auto* ipc = loop->enqueueIPC([callback](DeviceLoop* loop, volatile request::Request* req) {
            callback(loop, req->u.closeResult.error);
        });
        closeImpl(loop, ipc, ipc->request()->u.close, fd);
    }
    END_SINGLE_THREAD
}

template<typename Lambda>
inline __device__ auto ftruncate(DeviceLoop* loop, int fd, off_t offset, Lambda callback) -> void
{
    BEGIN_SINGLE_THREAD
    {
        auto* ipc = loop->enqueueIPC([callback](DeviceLoop* loop, volatile request::Request* req) {
            callback(loop, req->u.ftruncateResult.error);
        });
        ftruncateImpl(loop, ipc, ipc->request()->u.ftruncate, fd, offset);
    }
    END_SINGLE_THREAD
}

template<typename Lambda>
inline __device__ auto readOnePage(DeviceLoop* loop, int fd, size_t offset, size_t count, Lambda callback) -> void
{
    loop->allocOnePage([=](DeviceLoop* loop, volatile request::Request* req) {
        BEGIN_SINGLE_THREAD
        {
            void* page = req->u.allocOnePageResult.page;
            auto* ipc = loop->enqueueIPC([=](DeviceLoop* loop, volatile request::Request* req) {
                volatile request::Request oneTimeRequest;
                oneTimeRequest.u.readOnePageResult.page = page;
                oneTimeRequest.u.readOnePageResult.readCount = req->u.readResult.readCount;
                __threadfence();
                callback(loop, &oneTimeRequest);
            });
            readImpl(loop, ipc, ipc->request()->u.read, fd, offset, count, static_cast<unsigned char*>(page));
        }
        END_SINGLE_THREAD
    });
}

template<typename Lambda>
inline __device__ auto performOnePageRead(DeviceLoop* loop, int fd, size_t offset, size_t count, unsigned char* buffer, size_t requestedOffset, volatile request::Request* req, Lambda callback) -> void
{
    ssize_t readCount = req->u.readOnePageResult.readCount;
    void* page = req->u.readOnePageResult.page;
    ssize_t cursor = requestedOffset + readCount;
    ssize_t last = offset + count;

    GPU_ASSERT(readCount <= count);
    GPU_ASSERT(cursor <= last);
    if (readCount < 0) {
        callback(loop, -1);
        return;
    }

    if (cursor != last) {
        readOnePage(loop, fd, cursor, min((last - cursor), GLOOP_SHARED_PAGE_SIZE), [=](DeviceLoop* loop, volatile request::Request* req) {
            performOnePageRead(loop, fd, offset, count, buffer, cursor, req, callback);
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
    readOnePage(loop, fd, offset, min(count, GLOOP_SHARED_PAGE_SIZE), [=](DeviceLoop* loop, volatile request::Request* req) {
        performOnePageRead(loop, fd, offset, count, buffer, offset, req, callback);
    });
}

template<typename Lambda>
inline __device__ auto writeOnePage(DeviceLoop* loop, int fd, size_t offset, size_t transferringSize, unsigned char* buffer, Lambda callback) -> void
{
    loop->allocOnePage([=](DeviceLoop* loop, volatile request::Request* req) {
        unsigned char* page = static_cast<unsigned char*>(req->u.allocOnePageResult.page);
        gpunet::copy_block_dst_volatile(reinterpret_cast<volatile uchar*>(page), buffer, transferringSize);
        BEGIN_SINGLE_THREAD
        {
            auto* ipc = loop->enqueueIPC([=](DeviceLoop* loop, volatile request::Request* req) {
                BEGIN_SINGLE_THREAD
                {
                    loop->freeOnePage(page);
                }
                END_SINGLE_THREAD
                volatile request::Request oneTimeRequest;
                oneTimeRequest.u.writeOnePageResult.writtenCount = req->u.writeResult.writtenCount;
                callback(loop, &oneTimeRequest);
            });
            writeImpl(loop, ipc, ipc->request()->u.write, fd, offset, transferringSize, page);
        }
        END_SINGLE_THREAD
    });
}

template<typename Lambda>
inline __device__ auto performOnePageWrite(DeviceLoop* loop, int fd, size_t offset, size_t count, unsigned char* buffer, size_t requestedOffset, volatile request::Request* req, Lambda callback) -> void
{
    ssize_t writtenCount = req->u.writeOnePageResult.writtenCount;
    ssize_t cursor = requestedOffset + writtenCount;
    ssize_t last = offset + count;

    GPU_ASSERT(writtenCount <= count);
    GPU_ASSERT(cursor <= last);
    if (writtenCount < 0) {
        callback(loop, -1);
        return;
    }

    if (cursor != last) {
        writeOnePage(loop, fd, cursor, min((last - cursor), GLOOP_SHARED_PAGE_SIZE), buffer + (cursor - offset), [=](DeviceLoop* loop, volatile request::Request* req) {
            performOnePageWrite(loop, fd, offset, count, buffer, cursor, req, callback);
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
    writeOnePage(loop, fd, offset, min(count, GLOOP_SHARED_PAGE_SIZE), buffer, [=](DeviceLoop* loop, volatile request::Request* req) {
        performOnePageWrite(loop, fd, offset, count, buffer, offset, req, callback);
    });
}

template<typename Lambda>
inline __device__ auto mmap(DeviceLoop* loop, void* address, size_t size, int prot, int flags, int fd, off_t offset, Lambda callback) -> void
{
    BEGIN_SINGLE_THREAD
    {
        auto* ipc = loop->enqueueIPC([callback](DeviceLoop* loop, volatile request::Request* req) {
            callback(loop, req->u.mmapResult.address);
        });
        mmapImpl(loop, ipc, ipc->request()->u.mmap, address, size, prot, flags, fd, offset);
    }
    END_SINGLE_THREAD
}


template<typename Lambda>
inline __device__ auto munmap(DeviceLoop* loop, volatile void* address, size_t size, Lambda callback) -> void
{
    BEGIN_SINGLE_THREAD
    {
        auto* ipc = loop->enqueueIPC([callback](DeviceLoop* loop, volatile request::Request* req) {
            callback(loop, req->u.munmapResult.error);
        });
        munmapImpl(loop, ipc, ipc->request()->u.munmap, address, size);
    }
    END_SINGLE_THREAD
}

template<typename Lambda>
inline __device__ auto msync(DeviceLoop* loop, volatile void* address, size_t size, int flags, Lambda callback) -> void
{
    BEGIN_SINGLE_THREAD
    {
        auto* ipc = loop->enqueueIPC([callback](DeviceLoop* loop, volatile request::Request* req) {
            callback(loop, req->u.msyncResult.error);
        });
        msyncImpl(loop, ipc, ipc->request()->u.msync, address, size, flags);
    }
    END_SINGLE_THREAD
}

} }  // namespace gloop::fs
#endif  // GLOOP_FS_H_
