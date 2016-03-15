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

#include "data_log.h"
#include "monitor.h"
#include "monitor_server.h"
#include "monitor_session.h"
namespace gloop {
namespace monitor {

Server::Server(Monitor& monitor, uint32_t serverId)
    : m_monitor(monitor)
    , m_id(serverId)
    , m_ioService(monitor.ioService())
    , m_acceptor(m_ioService, boost::asio::local::stream_protocol::endpoint(Session::createName(GLOOP_ENDPOINT, serverId)))
{
    accept();
}

void Server::accept()
{
    auto* session = new Session(*this, m_monitor.nextId());
    m_acceptor.async_accept(session->socket(), [=](const boost::system::error_code& error) {
        if (!error) {
            session->handShake();
        } else {
            delete session;
        }
        accept();
    });
}

void Server::registerSession(Session& session)
{
    m_sessionList.push_back(session);
}

void Server::unregisterSession(Session& session)
{
    m_sessionList.erase(SessionList::s_iterator_to(session));
}

Session* Server::calculateNextSession()
{
    Session* target = nullptr;
    for (auto& session : sessionList()) {
        if (session.isAttemptingToLaunch()) {
            GLOOP_DEBUG("  candidate[%u], ticks:(%llu)\n", session.id(), (long long unsigned)session.used().count());
            if (!target) {
                target = &session;
            } else {
                if (target->used() > session.used()) {
                    target = &session;
                }
            }
        }
    }
#if 0
    // No scheduling.
    m_toBeAllowed = anySessionAllowed();
#else
    if (!target) {
        m_toBeAllowed = anySessionAllowed();
    } else {
        m_toBeAllowed = target->id();
    }
    GLOOP_DEBUG("Next ID[%u]\n", m_toBeAllowed);
    return target;
#endif
}

} }  // namsepace gloop::monitor
