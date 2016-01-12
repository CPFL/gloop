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
#include <gpufs/libgpufs/fs_calls.cu.h>
#include "device_loop.cuh"
#include "serialized.cuh"
namespace gloop {

__device__ DeviceLoop::DeviceLoop(uint64_t* buffer, size_t size)
    : m_buffer(buffer)
    , m_put(m_buffer)
    , m_get(m_buffer)
    , m_size(size)
    , m_index(0)
{
}

__device__ void* DeviceLoop::dequeue()
{
    __shared__ void* result;
    BEGIN_SINGLE_THREAD
    {
        size_t size = *m_get++;
        result = m_get;
        m_get += size;
        --m_index;
        GPU_ASSERT(m_put + m_size > m_get)
    }
    END_SINGLE_THREAD
    return result;
}

__device__ bool DeviceLoop::drain()
{
    while (!done()) {
        Serialized* lambda = reinterpret_cast<Serialized*>(dequeue());
        lambda->m_lambda(this, lambda->m_value);
    }
    return true;
}

}  // namespace gloop
