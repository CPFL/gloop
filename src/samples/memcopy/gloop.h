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
#ifndef GLOOP_H_
#define GLOOP_H_
#include <type_traits>
#include <utility>
#include "fs_calls.cu.h"
#include <gloop/gloop.h>

namespace gloop {

template<typename Callback>
struct Serialized {
    uint64_t m_value;
    Lambda<Callback> m_lambda;
};

template<>
struct Serialized<void> {
    uint64_t m_value;

    __device__ uint64_t value()
    {
        return m_value;
    }

    __device__ Callback& callback()
    {
        // The alignment of the m_value is largest in the system, 8byte.
        return *reinterpret_cast<Callback*>((reinterpret_cast<char*>(this) + sizeof(m_value)));
    }
};

template<typename Callback>
__device__ Serialized<Callback> makeSerialized(const Callback& callback, int value)
{
    return { value, Lambda<Callback>(callback) };
}

template<typename Callback>
__device__ auto open(DeviceLoop* loop, char* filename, int mode, Callback callback) -> void
{
    int fd = gopen(filename, mode);
    loop->enqueue(makeSerialized(callback, fd));
}

template<typename Callback>
__device__ auto write(DeviceLoop* loop, int fd, size_t offset, size_t count, unsigned char* buffer, Callback callback) -> void
{
    size_t written_size = gwrite(fd, offset, count, buffer);
    loop->enqueue(makeSerialized(callback, written_size));
}

template<typename Callback>
__device__ auto fstat(DeviceLoop* loop, int fd, Callback callback) -> void
{
    size_t value = ::fstat(fd);
    loop->enqueue(makeSerialized(callback, value));
}

template<typename Callback>
__device__ auto close(DeviceLoop* loop, int fd, Callback callback) -> void
{
    int err = gclose(fd);
    loop->enqueue(makeSerialized(callback, err));
}

template<typename Callback>
__device__ auto read(DeviceLoop* loop, int fd, size_t offset, size_t size, unsigned char* buffer, Callback callback) -> void
{
    size_t bytes_read = gread(fd, offset, size, buffer);
    loop->enqueue(makeSerialized(callback, bytes_read));
}

template<typename Callback, class... Args>
__global__ void launch(const Callback& callback, Args... args)
{
    uint64_t buffer[1024];
    DeviceLoop loop(buffer, 1024);
    callback(&loop, std::forward<Args>(args)...);
    while (!loop.done()) {
        Serialized<void>* lambda = reinterpret_cast<Serialized<void>*>(loop.dequeue());
        lambda->callback()(&loop, lambda->value());
    }
}

}  // namespace gloop
#endif  // GLOOP_H_
