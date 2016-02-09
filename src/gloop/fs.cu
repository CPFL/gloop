/*
  Copyright (C) 2016 Yusuke Suzuki <yusuke.suzuki@sslab.ics.keio.ac.jp>

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

#include "code.cuh"
#include "device_loop.cuh"
#include "fs.cuh"
#include "memcpy_io.cuh"
#include "request.h"

namespace gloop {
namespace fs {

__device__ void openImpl(DeviceLoop* loop, IPC* ipc, volatile request::Open& req, const char* filename, int mode)
{
    GLOOP_ASSERT_SINGLE_THREAD();
    memcpyIO(req.filename.data, filename, GLOOP_FILENAME_SIZE);
    req.mode = mode;
    ipc->emit(Code::Open);
}

__device__ void writeImpl(DeviceLoop* loop, IPC* ipc, volatile request::Write& req, int fd, size_t offset, size_t count, unsigned char* buffer)
{
    GLOOP_ASSERT_SINGLE_THREAD();
    req.fd = fd;
    req.offset = offset;
    req.count = count;
    req.buffer = buffer;
    ipc->emit(Code::Write);
}

__device__ void fstatImpl(DeviceLoop* loop, IPC* ipc, volatile request::Fstat& req, int fd)
{
    GLOOP_ASSERT_SINGLE_THREAD();
    req.fd = fd;
    ipc->emit(Code::Fstat);
}

__device__ void closeImpl(DeviceLoop* loop, IPC* ipc, volatile request::Close& req, int fd)
{
    GLOOP_ASSERT_SINGLE_THREAD();
    req.fd = fd;
    ipc->emit(Code::Close);
}

__device__ void readImpl(DeviceLoop* loop, IPC* ipc, volatile request::Read& req, int fd, size_t offset, size_t count, unsigned char* buffer)
{
    GLOOP_ASSERT_SINGLE_THREAD();
    req.fd = fd;
    req.offset = offset;
    req.count = count;
    req.buffer = buffer;
    ipc->emit(Code::Read);
}

__device__ void ftruncateImpl(DeviceLoop* loop, IPC* ipc, volatile request::Ftruncate& req, int fd, off_t offset)
{
    GLOOP_ASSERT_SINGLE_THREAD();
    req.fd = fd;
    req.offset = offset;
    ipc->emit(Code::Ftruncate);
}

__device__ void mmapImpl(DeviceLoop* loop, IPC* ipc, volatile request::Mmap& req, void* address, size_t size, int prot, int flags, int fd, off_t offset)
{
    GLOOP_ASSERT_SINGLE_THREAD();
    req.address = address;
    req.size = size;
    req.prot = prot;
    req.flags = flags;
    req.fd = fd;
    req.offset = offset;
    ipc->emit(Code::Mmap);
}

__device__ void munmapImpl(DeviceLoop* loop, IPC* ipc, volatile request::Munmap& req, volatile void* address, size_t size)
{
    GLOOP_ASSERT_SINGLE_THREAD();
    req.address = address;
    req.size = size;
    ipc->emit(Code::Munmap);
}

__device__ void msyncImpl(DeviceLoop* loop, IPC* ipc, volatile request::Msync& req, volatile void* address, size_t size, int flags)
{
    GLOOP_ASSERT_SINGLE_THREAD();
    req.address = address;
    req.size = size;
    req.flags = flags;
    ipc->emit(Code::Msync);
}

} }  // namespace gloop::fs
