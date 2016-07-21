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

#pragma once

#include <chrono>
#include <iostream>
namespace gloop {

class Benchmark {
public:
    typedef std::chrono::high_resolution_clock Clock;
    inline Clock::time_point begin()
    {
        m_begin = Clock::now();
        return m_begin;
    }

    inline Clock::time_point end()
    {
        m_end = Clock::now();
        return m_end;
    }

    std::chrono::microseconds ticks()
    {
        return std::chrono::duration_cast<std::chrono::microseconds>(m_end - m_begin);
    }

    void report(const std::string& prefix = "")
    {
        std::cout << prefix << "result:us(" << ticks().count() << ")" << std::endl;
    }

    void report(FILE* file, const std::string& prefix = "")
    {
        std::fprintf(file, "%sresult:us(%lld)\n", prefix.c_str(), (long long)ticks().count());
    }

    inline Clock::time_point beginPoint() const { return m_begin; }
    inline Clock::time_point endPoint() const { return m_end; }

private:
    Clock::time_point m_begin { };
    Clock::time_point m_end { };
};

typedef Benchmark TimeWatch;

}  // namespace gloop
