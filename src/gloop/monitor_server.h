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
#include <boost/thread/mutex.hpp>
#include "noncopyable.h"
#include "spinlock.h"
namespace gloop {
namespace monitor {

class Server {
GLOOP_NONCOPYABLE(Server);
public:
    // typedef boost::mutex Lock;
    typedef Spinlock Lock;

    Server(boost::asio::io_service& ioService, const char* endpoint);

    boost::asio::io_service& ioService() { return m_ioService; }
    const boost::asio::io_service& ioService() const { return m_ioService; }

    Lock& mutex() { return m_mutex; }

private:
    void accept();

    std::atomic<uint32_t> m_nextId { 0 };
    Lock m_mutex;
    boost::asio::io_service& m_ioService;
    boost::asio::local::stream_protocol::acceptor m_acceptor;
};

} }  // namsepace gloop::monitor
#endif  // GLOOP_MONITOR_SERVER_H_
