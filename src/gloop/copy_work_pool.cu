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
#include <mutex>
#include <thread>
#include "copy_work_pool.cuh"
namespace gloop {

CopyWork* CopyWorkPool::acquire()
{
    boost::unique_lock<boost::mutex> lock(m_mutex);
    while (m_works.empty()) {
        m_conditionVariable.wait(lock);
    }
    CopyWork* work = m_works.back();
    m_works.pop_back();
    return work;
}

CopyWork* CopyWorkPool::tryAcquire()
{
    boost::unique_lock<boost::mutex> lock(m_mutex);
    if (m_works.empty()) {
        return nullptr;
    }
    CopyWork* work = m_works.back();
    m_works.pop_back();
    return work;
}

void CopyWorkPool::release(CopyWork* work)
{
    boost::unique_lock<boost::mutex> lock(m_mutex);
    m_works.push_back(work);
    m_conditionVariable.notify_one();
}

void CopyWorkPool::registerCopyWork(std::shared_ptr<CopyWork> work)
{
    boost::unique_lock<boost::mutex> lock(m_mutex);
    m_holding.push_back(work);
    m_works.push_back(work.get());
}

}  // namespace gloop
