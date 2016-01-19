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
#include "request.cuh"

namespace gloop {
namespace fs {

__device__ void openImpl(DeviceLoop* loop, request::Open& req, char* filename, int mode)
{
    BEGIN_SINGLE_THREAD
    {
        memcpy(req.filename.data, filename, GLOOP_FILENAME_SIZE);
        req.mode = mode;
        loop->emit(Code::Open, reinterpret_cast<request::Request*>(&req));
    }
    END_SINGLE_THREAD
}

__device__ void writeImpl(DeviceLoop* loop, request::Write& req, int fd, size_t offset, size_t count, unsigned char* buffer)
{
    BEGIN_SINGLE_THREAD
    {
        req.fd = fd;
        req.offset = offset;
        req.count = count;
        req.buffer = buffer;
        loop->emit(Code::Write, reinterpret_cast<request::Request*>(&req));
    }
    END_SINGLE_THREAD
}

__device__ void fstatImpl(DeviceLoop* loop, request::Fstat& req, int fd)
{
    BEGIN_SINGLE_THREAD
    {
        req.fd = fd;
        loop->emit(Code::Fstat, reinterpret_cast<request::Request*>(&req));
    }
    END_SINGLE_THREAD
}

__device__ void closeImpl(DeviceLoop* loop, request::Close& req, int fd)
{
    BEGIN_SINGLE_THREAD
    {
        req.fd = fd;
        loop->emit(Code::Close, reinterpret_cast<request::Request*>(&req));
    }
    END_SINGLE_THREAD
}

__device__ void readImpl(DeviceLoop* loop, request::Read& req, int fd, size_t offset, size_t size, unsigned char* buffer)
{
    BEGIN_SINGLE_THREAD
    {
        req.fd = fd;
        req.offset = offset;
        req.size = size;
        req.buffer = buffer;
        loop->emit(Code::Read, reinterpret_cast<request::Request*>(&req));
    }
    END_SINGLE_THREAD
}

} }  // namespace gloop::fs
