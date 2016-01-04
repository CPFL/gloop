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

#include <atomic>
#include <boost/asio.hpp>
#include <unistd.h>
#include "config.h"
#include "monitor_session.h"

class Server {
public:
    Server(boost::asio::io_service& ioService, const char* endpoint);

private:
    void accept();

    std::atomic<uint32_t> m_nextId { 0 };
    boost::asio::io_service& m_ioService;
    boost::asio::local::stream_protocol::acceptor m_acceptor;
};


Server::Server(boost::asio::io_service& ioService, const char* endpoint)
    : m_ioService(ioService)
    , m_acceptor(ioService, boost::asio::local::stream_protocol::endpoint(endpoint))
{
    accept();
}

void Server::accept()
{
    auto* session = new gloop::monitor::Session(m_ioService, m_nextId++);
    m_acceptor.async_accept(session->socket(), [=](const boost::system::error_code& error) {
        if (!error) {
            session->handShake();
        } else {
            delete session;
        }
        accept();
    });
}

int main(int argc, char** argv)
{
    ::unlink(GLOOP_ENDPOINT);
    try {
        boost::asio::io_service ioService;
        Server server(ioService, GLOOP_ENDPOINT);
        ioService.run();
    } catch (std::exception& e) {
        fprintf(stderr, "Exception: %s\n", e.what());
    }

    return 0;
}
