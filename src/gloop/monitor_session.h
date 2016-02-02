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
#ifndef GLOOP_MONITOR_SESSION_H_
#define GLOOP_MONITOR_SESSION_H_
#include <boost/asio.hpp>
#include <boost/interprocess/ipc/message_queue.hpp>
#include <boost/thread.hpp>
#include <type_traits>
#include <memory>
#include <mutex>
#include "command.h"
#include "config.h"
#include "monitor_server.h"
#include "noncopyable.h"
namespace gloop {
namespace monitor {

class Session {
GLOOP_NONCOPYABLE(Session);
public:
    Session(Server&, uint32_t id);
    ~Session();

    boost::asio::local::stream_protocol::socket& socket() { return m_socket; }
    uint32_t id() const { return m_id; }

    typedef std::aligned_storage<sizeof(Command), std::alignment_of<Command>::value>::type CommandBuffer;

    void handShake();
    static std::string createName(const char* prefix, uint32_t id);
    static std::unique_ptr<boost::interprocess::message_queue> createQueue(const char* prefix, uint32_t id, bool create);

private:
    Command* buffer() { return reinterpret_cast<Command*>(&m_buffer); }

    void main();
    bool handle(Command&);
    void handleRead(const boost::system::error_code& error);
    void handleWrite(const boost::system::error_code& error);

    bool initialize(Command&);

    uint32_t m_id;
    Server& m_server;
    boost::asio::local::stream_protocol::socket m_socket;
    CommandBuffer m_buffer;
    std::unique_ptr<boost::thread> m_thread;
    std::unique_ptr<boost::interprocess::message_queue> m_mainQueue;
    std::unique_ptr<boost::interprocess::message_queue> m_requestQueue;
    std::unique_ptr<boost::interprocess::message_queue> m_responseQueue;
    std::unique_lock<Server::Lock> m_lock;
};

} }  // namsepace gloop::monitor
#endif  // GLOOP_MONITOR_SESSION_H_
