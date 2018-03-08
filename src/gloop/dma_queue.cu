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

#include "dma_queue.cuh"
#include "utility.cuh"

namespace gloop {

DMAQueue::DMAQueue()
{
    GLOOP_CUDA_SAFE_CALL(cudaStreamCreate(&m_stream));
    m_thread = boost::thread([&] {
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
            consume(std::move(pending));
            if (finalizing) {
                return;
            }
        }
    });
}

DMAQueue::~DMAQueue()
{
    {
        boost::unique_lock<boost::mutex> lock(m_mutex);
        m_finalizing = true;
        m_condition.notify_one();
    }
    m_thread.join();
    // cudaStreamDestroy(m_stream);
}

void DMAQueue::consume(std::deque<DMA>&& queue)
{
}

void DMAQueue::enqueue(Callback)
{
}


} // namespace gloop
