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
#ifndef GLOOP_SPINLOCK_H_
#define GLOOP_SPINLOCK_H_
#include <atomic>
#include "noncopyable.h"
namespace gloop {

// spinlock. Not considering thundering herd etc.
// http://stackoverflow.com/questions/26583433/c11-implementation-of-spinlock-using-atomic
class Spinlock {
GLOOP_NONCOPYABLE(Spinlock)
public:
    Spinlock() = default;

    void lock()
    {
        while (m_locked.test_and_set(std::memory_order_acquire));
    }

    void unlock()
    {
        m_locked.clear(std::memory_order_release);
    }

    bool try_lock()
    {
        return !m_locked.test_and_set(std::memory_order_acquire);
    }

private:
    std::atomic_flag m_locked { ATOMIC_FLAG_INIT };
};

}  // namespace gloop
#endif  // GLOOP_SPINLOCK_H_
