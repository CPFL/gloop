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

#include <cassert>
#include <gpufs/libgpufs/util.cu.h>
#include "gipc.cuh"

namespace gipc {

__device__ void Channel::emit()
{
    lock();
    {
        GPU_ASSERT(m_status == Wait);
        __threadfence_system();
        m_status = Status::Emit;
        __threadfence_system();
        WAIT_ON_MEM_NE(m_status, Status::Wait);
    }
    unlock();
}

__host__ void Channel::stop()
{
    __sync_synchronize();
    m_status = Status::Emit;
    __sync_synchronize();
}

__host__ void Channel::wait()
{
    while (m_status == Status::Wait);
    __sync_synchronize();
    m_status = Status::Wait;
    __sync_synchronize();
}

__device__ __host__ bool Channel::peek()
{
    return m_status == Status::Emit;
}

__device__ void Channel::lock()
{
    MUTEX_LOCK(m_lock);
}

__device__ void Channel::unlock()
{
    MUTEX_UNLOCK(m_lock);
}

}  // namespace gpic
