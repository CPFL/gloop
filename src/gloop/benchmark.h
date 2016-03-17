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
#ifndef GLOOP_BENCHMARK_H_
#define GLOOP_BENCHMARK_H_
#include <chrono>
#include <iostream>
namespace gloop {

class Benchmark {
public:
    typedef std::chrono::high_resolution_clock clock;
    inline void begin()
    {
        m_begin = clock::now();
    }

    inline void end()
    {
        m_end = clock::now();
    }

    std::chrono::microseconds ticks()
    {
        return std::chrono::duration_cast<std::chrono::microseconds>(m_end - m_begin);
    }

    void report(const std::string& prefix = "")
    {
        std::cout << prefix << "result:us(" << ticks().count() << ")" << std::endl;
    }

private:
    clock::time_point m_begin { };
    clock::time_point m_end { };
};

typedef Benchmark TimeWatch;

}  // namespace gloop
#endif  // GLOOP_BENCHMARK_H_
