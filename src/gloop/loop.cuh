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
#ifndef GLOOP_LOOP_CU_H_
#define GLOOP_LOOP_CU_H_
#include <utility>
#include "device_loop_inlines.cuh"
namespace gloop {
namespace loop {

template<typename DeviceLoop, typename Lambda>
inline __device__ auto postTask(DeviceLoop* loop, Lambda callback) -> void
{
    BEGIN_SINGLE_THREAD
    {
        loop->enqueueLater([callback](DeviceLoop* loop, volatile request::Request* req) {
            callback(loop);
        });
    }
    END_SINGLE_THREAD
}

template<typename DeviceLoop, typename Lambda>
inline __device__ auto forceExit(DeviceLoop* loop, Lambda callback) -> void
{
    BEGIN_SINGLE_THREAD
    {
        auto rpc = loop->enqueueRPC([callback](DeviceLoop* loop, volatile request::Request* req) {
            callback(loop);
        });
        rpc.emit(loop, Code::Exit);
    }
    END_SINGLE_THREAD
}

template<typename DeviceLoop, typename Lambda>
inline __device__ auto postTaskIfNecessary(DeviceLoop* loop, Lambda callback) -> int
{
    // CAUTION: Do not use shared memory to broadcast the result.
    // We use __syncthreads_or carefully here to scatter the boolean value.
    int posted = 0;
    __syncthreads();
    BEGIN_SINGLE_THREAD_WITHOUT_BARRIER
    {
        if (loop->shouldPostTask()) {
            posted = 1;
            loop->enqueueLater([callback](DeviceLoop* loop, volatile request::Request* req) {
                callback(loop);
            });
        }
    }
    END_SINGLE_THREAD_WITHOUT_BARRIER
    return __syncthreads_or(posted);
}


} }  // namespace gloop::loop
#endif  // GLOOP_LOOP_CU_H_
