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

#include <atomic>
#include <boost/asio.hpp>
#include <grpc++/grpc++.h>
#include "monitor_server.h"
#include "monitor_service.pb.h"
#include "monitor_service.grpc.pb.h"
#include "noncopyable.h"
namespace gloop {
namespace monitor {

#define GLOOP_MONITOR_ENDPOINT "/tmp/gloop_monitor_endpoint"

class Monitor final : private proto::Monitor::Service {
GLOOP_NONCOPYABLE(Monitor);
public:
    Monitor(uint32_t gpus, int enableUtilizationMonitorInMS);

    boost::asio::io_service& ioService() { return m_ioService; }

    void run();

    uint32_t nextId() { return m_nextId++; }

    bool enableUtilizationMonitor() const { return m_enableUtilizationMonitorInMS != 0; }
    int enableUtilizationMonitorInMS() const { return m_enableUtilizationMonitorInMS; }

private:
    grpc::Status listSessions(grpc::ServerContext* context, const proto::ListSessionRequest* request, grpc::ServerWriter<proto::Session>* writer) override;

    uint32_t m_gpus;
    int m_enableUtilizationMonitorInMS;
    boost::asio::io_service m_ioService;
    std::atomic<uint32_t> m_nextId { 0 };
    std::vector<std::shared_ptr<Server>> m_servers;
};

} }  // namsepace gloop::monitor
