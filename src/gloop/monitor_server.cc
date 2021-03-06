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
#include "data_log.h"
#include "make_unique.h"
#include "monitor_server.h"
#include "monitor_session.h"
#include "monitor_utility.h"
namespace gloop {
namespace monitor {

Server::Server(Monitor& monitor, uint32_t serverId)
    : m_monitor(monitor)
    , m_id(serverId)
    , m_ioService(monitor.ioService())
    , m_acceptor(m_ioService, boost::asio::local::stream_protocol::endpoint(createName(GLOOP_ENDPOINT, serverId)))
{
    if (m_monitor.enableUtilizationMonitor()) {
        m_utilizationAccounting = make_unique<UtilizationAccounting>(*this, m_monitor.enableUtilizationMonitorInMS());
        m_utilizationAccounting->start();
    }
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

static void setLowestSession(Session*& target, Session* given)
{
    if (!target) {
        target = given;
        return;
    }

    if (target->used() > given->used()) {
        target = given;
    }
}

void Server::progressCurrentVirtualTime(const std::lock_guard<Lock>&)
{
    Session* target = nullptr;
    for (auto& session : sessionList()) {
        setLowestSession(target, &session);
    }
    if (target) {
        // Burn the used. This always aligns CVT to 0.
        Session::Duration smallest = target->used();
        for (auto& session : sessionList()) {
            session.burnUsed(smallest);
            // GLOOP_DATA_LOG("  session[%u], ticks:(%llu)\n", session.id(), (long long unsigned)session.used().count());
        }
    }
}

void Server::registerSession(Session& newSession)
{
    std::lock_guard<Lock> guard(m_serverStatusLock);
    progressCurrentVirtualTime(guard);
    m_sessionList.push_back(newSession);
}

void Server::unregisterSession(Session& session)
{
    std::lock_guard<Lock> guard(m_serverStatusLock);
    m_sessionList.erase(SessionList::s_iterator_to(session));
    progressCurrentVirtualTime(guard);
}

Session* Server::calculateNextSession(const std::lock_guard<Lock>& locker)
{
    Session* target = nullptr;
    Session* lowestIncludingIO = nullptr;
    for (auto& session : sessionList()) {
        // GLOOP_DATA_LOG("  candidate[%u], ticks:(%lld)\n", session.id(), (long long int)session.used().count());
        if (session.isAttemptingToLaunch()) {
            setLowestSession(target, &session);
        }
        if (session.isAttemptingToLaunch() || !session.isScheduledDuringIO()) {
            setLowestSession(lowestIncludingIO, &session);
        }
    }
#if 0
    // No scheduling.
    m_toBeAllowed = anySessionAllowed();
#else
    if (!target) {
        m_toBeAllowed = anySessionAllowed();
    } else {
        // GLOOP_DATA_LOG("  selected candidate[%u], ticks:(%lld)\n", target->id(), (long long int)target->used().count());
        // Align to the least active session's time.
        Session::Duration smallest = target->used();
        for (auto& session : sessionList()) {
            session.burnUsed(smallest);
        }

        m_toBeAllowed = target->id();
        // If the lowestIncludingIO does not perform IO and sleep, it will be scheduled.
        // Adds the chance to boost when it is waken up.
        if (lowestIncludingIO && lowestIncludingIO != target) {
            lowestIncludingIO->setScheduledDuringIO();
            // GLOOP_DATA_LOG("  selected candidate[%u] IO marked\n", lowestIncludingIO->id());
        }
    }

    // GLOOP_DATA_LOG("Next ID[%u]\n", m_toBeAllowed);
    return target;
#endif
}
}
} // namsepace gloop::monitor
