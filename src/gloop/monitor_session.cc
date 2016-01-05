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
#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <mutex>
#include <vector>
#include "data_log.h"
#include "make_unique.h"
#include "monitor_session.h"

namespace gloop {
namespace monitor {

Session::Session(boost::asio::io_service& ioService, uint32_t id)
    : m_id(id)
    , m_socket(ioService)
{
}

Session::~Session()
{
    GLOOP_DATA_LOG("close:(%u)\n", static_cast<unsigned>(id()));
}

void Session::handShake()
{
    GLOOP_DATA_LOG("open:(%u)\n", static_cast<unsigned>(id()));
    boost::asio::async_read(m_socket, boost::asio::buffer(&m_buffer, sizeof(Command)), boost::bind(&Session::handleRead, this, boost::asio::placeholders::error));
}

void Session::handleRead(const boost::system::error_code& error)
{
    if (error) {
        delete this;
        return;
    }
    Command command(*buffer());
    this->handle(command);
    // handle command
    boost::asio::async_write(m_socket, boost::asio::buffer(&m_buffer, sizeof(Command)), boost::bind(&Session::handleWrite, this, boost::asio::placeholders::error));
}

void Session::handleWrite(const boost::system::error_code& error)
{
    if (error) {
        delete this;
        return;
    }

    boost::asio::async_read(m_socket, boost::asio::buffer(&m_buffer, sizeof(Command)), boost::bind(&Session::handleRead, this, boost::asio::placeholders::error));
}

bool Session::handle(Command& command)
{
    switch (command.type) {
    case Command::Type::Initialize:
        return initialize(command);
    case Command::Type::Operation:
        break;
    }
    return false;
}

bool Session::initialize(Command& command)
{
    m_requestQueue = Session::createQueue(GLOOP_SHARED_REQUEST_QUEUE, id(), true);
    m_responseQueue = Session::createQueue(GLOOP_SHARED_RESPONSE_QUEUE, id(), true);
    m_thread = make_unique<boost::thread>(&Session::main, this);

    command = (Command) {
        .type = Command::Type::Initialize,
        .payload = id()
    };
    return true;
}

std::unique_ptr<boost::interprocess::message_queue> Session::createQueue(const char* prefix, uint32_t id, bool create)
{
    std::vector<char> name(200);
    const int ret = std::snprintf(name.data(), name.size() - 1, "%s%u", prefix, id);
    if (ret < 0) {
        std::perror(nullptr);
        std::exit(1);
    }
    name[ret] = '\0';
    if (create) {
        boost::interprocess::message_queue::remove(name.data());
        return make_unique<boost::interprocess::message_queue>(boost::interprocess::create_only, name.data(), 0x100000, sizeof(Command));
    }
    return make_unique<boost::interprocess::message_queue>(boost::interprocess::open_only, name.data());
}

void Session::main()
{
    while (true) {
        unsigned int priority { };
        std::size_t size { };
        Command command { };
        m_requestQueue->receive(&command, sizeof(Command), size, priority);
        if (handle(command)) {
            send(command);
        }
    }
}

void Session::send(Command command)
{
    m_responseQueue->send(&command, sizeof(Command), 0);
}

} }  // namsepace gloop::monitor
