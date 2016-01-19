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
#include <gpufs/libgpufs/fs_calls.cu.h>
#include <type_traits>
#include <utility>
#include "device_loop.cuh"
#include "request.cuh"

namespace gloop {
namespace fs {

__device__ void openImpl(DeviceLoop*, request::Open&, char* filename, int mode);
__device__ void writeImpl(DeviceLoop*, request::Write&, int fd, size_t offset, size_t count, unsigned char* buffer);
__device__ void fstatImpl(DeviceLoop*, request::Fstat&, int fd);
__device__ void closeImpl(DeviceLoop*, request::Close&, int fd);
__device__ void readImpl(DeviceLoop*, request::Read&, int fd, size_t offset, size_t size, unsigned char* buffer);

template<typename Lambda>
inline __device__ auto open(DeviceLoop* loop, char* filename, int mode, Lambda callback) -> void
{
    auto* req = loop->enqueue([callback](DeviceLoop* loop, request::Request* req) {
        callback(loop, req->u.result.result);
    });
    openImpl(loop, req->u.open, filename, mode);
}

template<typename Lambda>
inline __device__ auto write(DeviceLoop* loop, int fd, size_t offset, size_t count, unsigned char* buffer, Lambda callback) -> void
{
    auto* req = loop->enqueue([callback](DeviceLoop* loop, request::Request* req) {
        callback(loop, req->u.result.result);
    });
    writeImpl(loop, req->u.write, fd, offset, count, buffer);
}

template<typename Lambda>
inline __device__ auto fstat(DeviceLoop* loop, int fd, Lambda callback) -> void
{
    auto* req = loop->enqueue([callback](DeviceLoop* loop, request::Request* req) {
        callback(loop, req->u.result.result);
    });
    fstatImpl(loop, req->u.fstat, fd);
}

template<typename Lambda>
inline __device__ auto close(DeviceLoop* loop, int fd, Lambda callback) -> void
{
    auto* req = loop->enqueue([callback](DeviceLoop* loop, request::Request* req) {
        callback(loop, req->u.result.result);
    });
    closeImpl(loop, req->u.close, fd);
}

template<typename Lambda>
inline __device__ auto read(DeviceLoop* loop, int fd, size_t offset, size_t size, unsigned char* buffer, Lambda callback) -> void
{
    auto* req = loop->enqueue([callback](DeviceLoop* loop, request::Request* req) {
        callback(loop, req->u.result.result);
    });
    readImpl(loop, req->u.read, fd, offset, size, buffer);
}

} }  // namespace gloop::fs
#endif  // GLOOP_FS_H_
