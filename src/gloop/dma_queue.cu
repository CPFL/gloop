/*
  Copyright (C) 2018 Yusuke Suzuki <yusuke.suzuki@sslab.ics.keio.ac.jp>

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

#pragma once

#include "copy_work.cuh"
#include "data_log.h"
#include "dma_queue.cuh"
#include "host_loop.cuh"
#include "utility.cuh"
#include <cuda.h>
#include <mutex>

namespace gloop {

DMAQueue::DMAQueue(HostLoop& hostLoop)
    : m_hostLoop(hostLoop)
{
    GLOOP_CUDA_SAFE_CALL(cudaStreamCreate(&m_stream));
    m_thread = boost::thread([&] {
        m_hostLoop.initializeInThread();
        while (true) {
            std::deque<DMA> pending;
            bool finalizing = false;
            {
                boost::unique_lock<boost::mutex> lock(m_mutex);
                while (m_queue.empty() && !m_finalizing) {
                    m_condition.wait(lock);
                }
                pending.swap(m_queue);
                finalizing = m_finalizing;
            }
            if (!pending.empty()) {
                consume(pending);
            }
            if (finalizing) {
                return;
            }
        }
    });
}

DMAQueue::~DMAQueue()
{
    {
        std::lock_guard<boost::mutex> lock(m_mutex);
        m_finalizing = true;
        m_condition.notify_one();
    }
    m_thread.join();
    // cudaStreamDestroy(m_stream);
}

void DMAQueue::consume(const std::deque<DMA>& queue)
{
    for (auto& dma : queue) {
        GLOOP_CUDA_SAFE_CALL(cudaMemcpyAsync(
            dma.memory(),
            dma.work()->hostMemory().hostPointer(),
            dma.size(),
            cudaMemcpyHostToDevice,
            m_stream));
    }
    GLOOP_CUDA_SAFE_CALL(cudaStreamSynchronize(m_stream));
    for (auto& dma : queue) {
        dma.callback()();
    }
}

void DMAQueue::enqueue(DMA dma)
{
    std::lock_guard<boost::mutex> lock(m_mutex);
    m_queue.push_back(dma);
    m_condition.notify_one();
}


} // namespace gloop
