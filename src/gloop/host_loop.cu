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
#include <boost/asio.hpp>
#include <cassert>
#include <cstdio>
#include <cuda_runtime_api.h>
#include "command.h"
#include "config.h"
#include "host_loop.cuh"
#include "make_unique.h"
#include "monitor_session.h"
namespace gloop {

HostLoop::HostLoop(volatile GPUGlobals* globals)
    : m_globals(globals)
    , m_loop(uv_loop_new())
    , m_socket(m_ioService)
{
    // Connect to the gloop monitor.
    {
        m_socket.connect(boost::asio::local::stream_protocol::endpoint(GLOOP_ENDPOINT));
        Command command = {
            .type = Command::Type::Initialize,
        };
        Command result { };
        while (true) {
            boost::system::error_code error;
            boost::asio::write(
                m_socket,
                boost::asio::buffer(reinterpret_cast<const char*>(&command), sizeof(Command)),
                boost::asio::transfer_all(),
                error);
            if (error != boost::asio::error::make_error_code(boost::asio::error::interrupted)) {
                break;
            }
            // retry
        }
        while (true) {
            boost::system::error_code error;
            boost::asio::read(
                m_socket,
                boost::asio::buffer(reinterpret_cast<char*>(&result), sizeof(Command)),
                boost::asio::transfer_all(),
                error);
            if (error != boost::asio::error::make_error_code(boost::asio::error::interrupted)) {
                break;
            }
        }
        m_id = result.payload;
    }
    m_requestQueue = monitor::Session::createQueue(GLOOP_SHARED_REQUEST_QUEUE, m_id, false);
    m_responseQueue = monitor::Session::createQueue(GLOOP_SHARED_RESPONSE_QUEUE, m_id, false);

    runPoller();
}

HostLoop::~HostLoop()
{
    uv_loop_close(m_loop);
    stopPoller();
}

// GPU RPC poller.
void HostLoop::runPoller()
{
    assert(!m_poller);
    m_stop.store(false, std::memory_order_release);
    m_poller = make_unique<std::thread>([this]() {
        pollerMain();
    });
}

void HostLoop::stopPoller()
{
    m_stop.store(true, std::memory_order_release);
    if (m_poller) {
        m_poller->join();
        m_poller.reset();
    }
}

void HostLoop::pollerMain()
{
    Command command = {
        .type = Command::Type::Operation,
        .payload = Command::Operation::Complete,
    };
    m_responseQueue->send(&command, sizeof(Command), 0);
    while (!m_stop.load(std::memory_order_acquire)) {
    }
}

void HostLoop::wait()
{
    while (true) {
        Command result = { };
        unsigned int priority { };
        std::size_t size { };
        m_responseQueue->receive(&result, sizeof(Command), size, priority);
        if (handle(result)) {
            break;
        }
    }
}

bool HostLoop::handle(Command command)
{
    return true;
}

}  // namespace gloop
