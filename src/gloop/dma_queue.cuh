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

#include <deque>
#include <boost/thread.hpp>
#include "noncopyable.h"

namespace gloop {

class CopyWork;
class HostLoop;

class DMAQueue {
public:
    using Callback = std::function<void()>;

    class DMA {
    public:
        DMA(CopyWork* work, void* memory, size_t size, Callback&& callback)
            : m_work(work)
            , m_memory(memory)
            , m_size(size)
            , m_callback(std::move(callback))
        {
        }

        void* memory() const
        {
            return m_memory;
        }

        size_t size() const
        {
            return m_size;
        }

        CopyWork* work() const
        {
            return m_work;
        }

        const Callback& callback() const
        {
            return m_callback;
        }

    private:
        CopyWork* m_work { nullptr };
        void* m_memory { nullptr };
        size_t m_size { 0 };
        Callback m_callback;
    };

    DMAQueue(HostLoop&);
    ~DMAQueue();

    cudaStream_t stream()
    {
        return m_stream;
    }

    void enqueue(DMA);

private:
    void consume(const std::deque<DMA>&);

    std::deque<DMA> m_queue;
    boost::mutex m_mutex;
    boost::condition_variable m_condition;
    boost::thread m_thread;
    cudaStream_t m_stream;
    HostLoop& m_hostLoop;
    bool m_finalizing { false };
};

} // namespace gloop
