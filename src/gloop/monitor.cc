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

#include "monitor.h"
#include "config.h"
#include "make_unique.h"
#include "monitor_server.h"
#include "monitor_service.grpc.pb.h"
#include "monitor_service.pb.h"
#include "monitor_session.h"
#include "monitor_utility.h"
#include <atomic>
#include <boost/thread.hpp>
#include <grpc++/grpc++.h>
#include <unistd.h>
namespace gloop {
namespace monitor {

Monitor::Monitor(uint32_t gpus, int enableUtilizationMonitorInMS)
    : m_gpus(gpus)
    , m_enableUtilizationMonitorInMS(enableUtilizationMonitorInMS)
{
    for (uint32_t i = 0; i < m_gpus; ++i) {
        ::unlink(createName(GLOOP_ENDPOINT, i).c_str());
    }
    ::unlink(GLOOP_MONITOR_ENDPOINT);
}

void Monitor::run()
{
    for (uint32_t i = 0; i < m_gpus; ++i) {
        m_servers.push_back(std::make_shared<Server>(*this, i));
    }

    grpc::ServerBuilder builder;
    std::string endpoint("unix:");
    endpoint.append(GLOOP_MONITOR_ENDPOINT);
    builder.AddListeningPort(endpoint, grpc::InsecureServerCredentials());
    builder.RegisterService(this);
    //     std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
    //     std::unique_ptr<boost::thread> monitoringThread = make_unique<boost::thread>([&] {
    //         server->Wait();
    //     });
    //
    m_ioService.run();
    // server->Shutdown();
}

grpc::Status Monitor::listSessions(grpc::ServerContext* context, const proto::ListSessionRequest* request, grpc::ServerWriter<proto::Session>* writer)
{
    for (auto& server : m_servers) {
        std::lock_guard<Lock> locker(server->serverStatusLock());
        for (auto& session : server->sessionList()) {
            proto::Session result;
            result.set_id(session.id());
            result.set_server(server->id());
            result.set_used(session.used().count());
            writer->Write(result);
        }
    }
    return grpc::Status::OK;
}

grpc::Status Monitor::listSwitchCount(grpc::ServerContext* context, const proto::SwitchCountRequest* request, grpc::ServerWriter<proto::SwitchCount>* writer)
{
    for (auto& server : m_servers) {
        std::lock_guard<Lock> locker(server->serverStatusLock());
        proto::SwitchCount result;
        result.set_server(server->id());
        result.set_value(server->switchCount());
        writer->Write(result);
    }
    return grpc::Status::OK;
}
}
} // namsepace gloop::monitor
