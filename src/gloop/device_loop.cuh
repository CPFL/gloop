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
#ifndef GLOOP_DEVICE_LOOP_H_
#define GLOOP_DEVICE_LOOP_H_
#include <cstdint>
#include "function.cuh"
#include "utility.h"
namespace gloop {

class DeviceLoop {
public:
    typedef gloop::function<void(DeviceLoop*, int)> Callback;

    __device__ DeviceLoop(uint64_t* buffer, size_t size);

    __device__ void enqueue(Callback lambda);

    __device__ void* dequeue();

    __device__ bool done();

    __device__ bool drain();

private:
    uint64_t* m_buffer;
    uint64_t* m_put;
    uint64_t* m_get;
    size_t m_size;
    size_t m_index;
};

inline __device__ bool DeviceLoop::done()
{
    __shared__ bool result;
    BEGIN_SINGLE_THREAD
    {
        result = m_index == 0;
    }
    END_SINGLE_THREAD
    return result;
}

}  // namespace gloop
#endif  // GLOOP_DEVICE_LOOP_H_
