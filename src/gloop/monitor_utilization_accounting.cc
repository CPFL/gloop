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

#include "benchmark.h"
#include "data_log.h"
#include "monitor_lock.h"
#include "monitor_server.h"
#include "monitor_utilization_accounting.h"
#include "make_unique.h"
#include <algorithm>
#include <mutex>
#include <thread>

namespace gloop {
namespace monitor {


UtilizationAccounting::UtilizationAccounting(Server& server, int epochInMS)
    : m_server(server)
    , m_epoch(std::chrono::milliseconds(epochInMS))
{
}

void UtilizationAccounting::start()
{
    m_thread = make_unique<boost::thread>([&] {
        TimeWatch watch;
        std::vector<std::pair<uint64_t, uint64_t>> data;
        uint64_t epoch = 0;
        while (true) {
            watch.begin();
            {
                data.clear();
                {
                    std::lock_guard<Lock> guard(m_server.serverStatusLock());
                    for (auto& session : m_server.sessionList()) {
                        data.push_back(std::make_pair(session.id(), session.readAndClearUtil()));
                    }
                }
                dump(epoch, data);
            }
            watch.end();
            auto sleepPeriod = m_epoch - watch.ticks();
            if (sleepPeriod > std::chrono::milliseconds(0)) {
                std::this_thread::sleep_for(sleepPeriod);
            }
            ++epoch;
        }
    });
}

void UtilizationAccounting::dump(uint64_t epoch, Data& data)
{
    std::sort(data.begin(), data.end());
}

} }  // namsepace gloop::monitor
