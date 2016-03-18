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
#ifndef GLOOP_MONITOR_SERVER_H_
#define GLOOP_MONITOR_SERVER_H_
#include <atomic>
#include <boost/asio.hpp>
#include <boost/intrusive/list.hpp>
#include <boost/thread.hpp>
#include <queue>
#include "monitor_lock.h"
#include "monitor_session.h"
#include "noncopyable.h"
#include "utility.h"
namespace gloop {
namespace monitor {

class Monitor;

struct SessionPriorityFunctor {
    bool operator()(Session* left, Session* right)
    {
        return left->used() > right->used();
    }
};

class Server {
GLOOP_NONCOPYABLE(Server);
public:
    typedef boost::intrusive::list<Session> SessionList;

    constexpr static uint32_t anySessionAllowed() { return UINT32_MAX; }

    Server(Monitor& monitor, uint32_t serverId);

    uint32_t id() const { return m_id; }

    boost::asio::io_service& ioService() { return m_ioService; }
    const boost::asio::io_service& ioService() const { return m_ioService; }

    ServerLock& kernelLock() { return m_kernelLock; }
    SessionList& sessionList() { return m_sessionList; }

    static std::string createEndpoint(const std::string& prefix, uint32_t id);

    void registerSession(Session&);
    void unregisterSession(Session&);

    bool isAllowed(Session& session) const;

    boost::condition_variable_any& condition() { return m_condition; }

    Session* calculateNextSession(std::lock_guard<Lock>&);
    Lock& serverStatusLock() { return m_serverStatusLock; }

private:
    void accept();

    void progressCurrentVirtualTime(std::lock_guard<Lock>&);

    Monitor& m_monitor;
    uint32_t m_id;
    std::atomic<uint32_t> m_waitingCount { 0 };
    ServerLock m_kernelLock;
    Lock m_serverStatusLock;
    SessionList m_sessionList;
    boost::asio::io_service& m_ioService;
    boost::asio::local::stream_protocol::acceptor m_acceptor;
    boost::condition_variable_any m_condition;
    uint32_t m_toBeAllowed { anySessionAllowed() };
    // std::priority_queue<Session*, std::vector<Session*>, SessionPriorityFunctor> m_priorityQueue;
};

GLOOP_ALWAYS_INLINE bool Server::isAllowed(Session& session) const
{
    return m_toBeAllowed == anySessionAllowed() || m_toBeAllowed == session.id();
}

} }  // namsepace gloop::monitor
#endif  // GLOOP_MONITOR_SERVER_H_
