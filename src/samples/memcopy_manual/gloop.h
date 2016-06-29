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
#ifndef GLOOP_H_
#define GLOOP_H_
#include <type_traits>
#include <utility>
#include "fs_calls.cu.h"
#include <gloop/device_loop.cuh>

enum LambdaId {
    Open1,
    Open2,
    Fstat1,
    Read1,
    Write1,
    Close1,
    Close2
};

__device__ void perform_copy(gloop::DeviceLoop<>* loop, uchar* scratch, int zfd, int zfd1, size_t me, size_t filesize);

struct Close2Data {
    LambdaId id;
};

__device__ void close2(gloop::DeviceLoop<>* loop, Close2Data* data);

struct Close1Data {
    LambdaId id;
    int zfd1;
};

__device__ void close1(gloop::DeviceLoop<>* loop, Close1Data* data);

struct Write1Data {
    LambdaId id;
    uchar* scratch;
    int zfd;
    int zfd1;
    size_t filesize;
    size_t me;
    size_t toRead;
    size_t written;
};

__device__ void write1(gloop::DeviceLoop<>* loop, Write1Data* data);

struct Read1Data {
    LambdaId id;
    uchar* scratch;
    int zfd;
    int zfd1;
    size_t filesize;
    size_t me;
    size_t toRead;
    size_t read;
};

__device__ void read1(gloop::DeviceLoop<>* loop, Read1Data* data);

struct Fstat1Data {
    LambdaId id;
    uchar* scratch;
    int zfd;
    int zfd1;
    size_t filesize;
};

__device__ void fstat1(gloop::DeviceLoop<>* loop, Fstat1Data* data);

struct Open2Data {
    LambdaId id;
    uchar* scratch;
    int zfd;
    int zfd1;
};

__device__ void open2(gloop::DeviceLoop<>* loop, Open2Data* data);

struct Open1Data {
    LambdaId id;
    uchar* scratch;
    char* dst;
    int zfd;
};

__device__ void open1(gloop::DeviceLoop<>* loop, Open1Data* data);

inline __device__ void store(Open1Data& data, int fd)
{
    data.zfd = fd;
}

inline __device__ void store(Open2Data& data, int fd)
{
    data.zfd1 = fd;
}

inline __device__ void store(Write1Data& data, size_t size)
{
    data.written = size;
}

inline __device__ void store(Read1Data& data, size_t size)
{
    data.read = size;
}

inline __device__ void store(Fstat1Data& data, size_t size)
{
    data.filesize = size;
}

namespace gloop {

template<typename Callback>
__device__ auto open(DeviceLoop* loop, char* filename, int mode, Callback callback) -> void
{
    int fd = gopen(filename, mode);
    store(callback, fd);
    loop->enqueue(callback);
    // return callback(fd);
}

template<typename Callback>
__device__ auto write(DeviceLoop* loop, int fd, size_t offset, size_t count, unsigned char* buffer, Callback callback) -> void
{
    size_t written_size = gwrite(fd, offset, count, buffer);
    store(callback, written_size);
    loop->enqueue(callback);
    // return callback(written_size);
}

template<typename Callback>
__device__ auto fstat(DeviceLoop* loop, int fd, Callback callback) -> void
{
    size_t value = ::fstat(fd);
    store(callback, value);
    loop->enqueue(callback);
    // return callback(value);
}

template<typename Callback>
__device__ auto close(DeviceLoop* loop, int fd, Callback callback) -> void
{
    int err = gclose(fd);
    loop->enqueue(callback);
    // return callback(err);
}

template<typename Callback>
__device__ auto read(DeviceLoop* loop, int fd, size_t offset, size_t size, unsigned char* buffer, Callback callback) -> void
{
    size_t bytes_read = gread(fd, offset, size, buffer);
    store(callback, bytes_read);
    loop->enqueue(callback);
    // return callback(bytes_read);
}

template<typename Callback, class... Args>
__global__ void launch(const Callback& callback, Args... args)
{
    uint64_t buffer[1024];
    DeviceLoop loop(buffer, 1024);
    callback(&loop, std::forward<Args>(args)...);
    while (!loop.done()) {
        void* lambda = loop.dequeue();
        switch (*(LambdaId*)(lambda)) {
        case Open1: {
            Open1Data* data = (Open1Data*)lambda;
            open1(&loop, data);
            break;
        }

        case Open2: {
            Open2Data* data = (Open2Data*)lambda;
            open2(&loop, data);
            break;
        }

        case Fstat1: {
            Fstat1Data* data = (Fstat1Data*)lambda;
            fstat1(&loop, data);
            break;
        }

        case Read1: {
            Read1Data* data = (Read1Data*)lambda;
            read1(&loop, data);
            break;
        }

        case Write1: {
            Write1Data* data = (Write1Data*)lambda;
            write1(&loop, data);
            break;
        }

        case Close1: {
            Close1Data* data = (Close1Data*)lambda;
            close1(&loop, data);
            break;
        }

        case Close2: {
            Close2Data* data = (Close2Data*)lambda;
            close2(&loop, data);
            break;
        }

        default:
            break;
        }
    }
}

}  // namespace gloop
#endif  // GLOOP_H_
