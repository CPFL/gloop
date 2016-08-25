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

#include "benchmark.h"
#include <array>

namespace gloop {

#define GLOOP_STATISTICS_TYPE(V) \
    V(IO, 0)                     \
    V(Kernel, 1)                 \
    V(Copy, 2)                   \
    V(DataInit, 3)               \
    V(GPUInit, 4)                \
    V(None, -1)

class Statistics {
public:
    enum class Type : int32_t {
#define GLOOP_LIST_ENUM(type, value) type = (value),
        GLOOP_STATISTICS_TYPE(GLOOP_LIST_ENUM)
#undef GLOOP_LIST_ENUM
    };
    static constexpr const size_t NumberOfTypes = 5;

    template <Type type>
    void begin()
    {
        begin(type);
    }

    template <Type type>
    void end()
    {
        end(type);
    }

    void begin(Type type)
    {
        m_benchmarks[static_cast<int32_t>(type)].begin();
    }

    void end(Type type)
    {
        gloop::Benchmark& benchmark = m_benchmarks[static_cast<int32_t>(type)];
        benchmark.end();
        m_times[static_cast<int32_t>(type)] += benchmark.ticks();
    }

    void report(FILE* file)
    {
#define GLOOP_REPORT(type, value)            \
    if (Type::type != Type::None) {          \
        report(file, Type::type, #type " "); \
    }
        GLOOP_STATISTICS_TYPE(GLOOP_REPORT)
#undef GLOOP_REPORT
    }

    template <Type type, typename Functor>
    void bench(Functor functor)
    {
        begin<type>();
        functor();
        end<type>();
    }

    static Statistics& instance()
    {
        static Statistics statistics;
        return statistics;
    }

    template <Type type>
    void switchTo()
    {
        switchTo(type);
    }

    void switchTo(Type type)
    {
        if (m_currentType != Type::None) {
            end(m_currentType);
        }
        if (type != Type::None) {
            begin(type);
        }
        m_currentType = type;
    }

    template <Type type>
    class Scope {
    public:
        Scope()
            : m_previousType(instance().currentType())
        {
            instance().switchTo(type);
        }

        ~Scope()
        {
            instance().switchTo(m_previousType);
        }

    private:
        Type m_previousType{Type::None};
    };

private:
    Type currentType() const
    {
        return m_currentType;
    }

    void report(FILE* file, Type type, const std::string& prefix = "")
    {
        std::fprintf(file, "%sresult:us(%lld)\n", prefix.c_str(), static_cast<long long>(m_times[static_cast<int32_t>(type)].count()));
    }

    Type m_currentType{Type::None};
    std::array<std::chrono::microseconds, NumberOfTypes> m_times{};
    std::array<gloop::Benchmark, NumberOfTypes> m_benchmarks{};
};

} // namespace gloop
