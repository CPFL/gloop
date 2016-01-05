/*
  Copyright (C) 2015 Yusuke Suzuki <yusuke.suzuki@sslab.ics.keio.ac.jp>

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
#ifndef GLOOP_HOST_LOOP_H_
#define GLOOP_HOST_LOOP_H_
#include <atomic>
#include <boost/asio.hpp>
#include <boost/interprocess/ipc/message_queue.hpp>
#include <thread>
#include <memory>
#include <uv.h>
#include "command.h"

struct GPUGlobals;

namespace boost {
class thread;
}  // namespace boost

namespace gloop {

class HostLoop {
public:
    HostLoop(volatile GPUGlobals*);
    ~HostLoop();

    uint32_t id() const { return m_id; }

    void wait();

private:
    void runPoller();
    void stopPoller();
    void pollerMain();

    bool handle(Command);

    volatile GPUGlobals* m_globals;
    uv_loop_t* m_loop;
    std::atomic<bool> m_stop { false };
    std::unique_ptr<boost::thread> m_poller;

    uint32_t m_id { 0 };
    boost::asio::io_service m_ioService;
    boost::asio::local::stream_protocol::socket m_socket;
    std::unique_ptr<boost::interprocess::message_queue> m_requestQueue;
    std::unique_ptr<boost::interprocess::message_queue> m_responseQueue;
};

}  // namespace gloop
#endif  // GLOOP_HOST_LOOP_H_
