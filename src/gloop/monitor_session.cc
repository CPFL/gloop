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
#include <boost/bind.hpp>
#include <chrono>
#include <mutex>
#include <vector>
#include "data_log.h"
#include "make_unique.h"
#include "monitor_server.h"
#include "monitor_session.h"
#include "monitor_utility.h"
#include "sync_read_write.h"

namespace gloop {
namespace monitor {

Session::Session(Server& server, uint32_t id)
    : m_id(id)
    , m_server(server)
    , m_socket(server.ioService())
    , m_kernelLock(server.kernelLock(), std::defer_lock)
    , m_timer(server.ioService())
{
}

Session::~Session()
{
    // NOTE: This destructor is always executed in single thread.
    m_server.unregisterSession(*this);
    // GLOOP_DATA_LOG("server:(%u),close:(%u)\n", m_server.id(), static_cast<unsigned>(id()));
    if (m_thread) {
        m_thread->interrupt();
        m_thread->join();
        m_thread.reset();
    }
}

void Session::handShake()
{
    // GLOOP_DATA_LOG("server:(%u),open:(%u)\n", m_server.id(), static_cast<unsigned>(id()));
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
    boost::asio::async_write(m_socket, boost::asio::buffer(&command, sizeof(Command)), boost::bind(&Session::handleWrite, this, boost::asio::placeholders::error));
}

void Session::handleWrite(const boost::system::error_code& error)
{
    if (error) {
        delete this;
        return;
    }

    boost::asio::async_read(m_socket, boost::asio::buffer(&m_buffer, sizeof(Command)), boost::bind(&Session::handleRead, this, boost::asio::placeholders::error));
}

void Session::kill()
{
    std::lock_guard<Lock> guard(m_lock);
    if (m_kernelLock.owns_lock()) {
        m_killed = true;
        m_killTimer.begin();
        syncWrite<uint32_t>(static_cast<volatile uint32_t*>(m_signal->get_address()), 1);
    }
}

void Session::configureTick(boost::asio::high_resolution_timer& timer)
{
    timer.expires_from_now(std::chrono::milliseconds(GLOOP_KILL_TIME));
    timer.async_wait([&](const boost::system::error_code& ec) {
        if (!ec) {
            // This is ASIO call. So it is executed under the main thread now. (Since only the main thread invokes ASIO's ioService.run()).
            for (auto& session : m_server.sessionList()) {
                if (&session != this) {
                    if (session.isAttemptingToLaunch()) {
                        // Found. Let's kill the current kernel executing.
                        kill();
                        break;
                    }
                }
            }
            configureTick(timer);
        }
    });
}

bool Session::handle(Command& command)
{
    switch (command.type) {
    case Command::Type::Initialize:
        return initialize(command);

    case Command::Type::Operation:
        return false;

    case Command::Type::Lock: {
        // GLOOP_DATA_LOG("[%u] Attempt to lock kernel token.\n", m_id);
        {
            {
                std::lock_guard<Lock> guard(m_lock);
                m_attemptToLaunch.store(true);

                // IO boosting.
                if (m_scheduledDuringIO) {
                    // auto polling = std::max<std::chrono::microseconds>(std::chrono::duration_cast<std::chrono::microseconds>(TimeWatch::Clock::now() - m_timeWatch.endPoint()), std::chrono::microseconds(0));
                    auto compensation = std::min<std::chrono::microseconds>(m_burned, std::chrono::duration_cast<std::chrono::microseconds>(boostThreshold()));
                    m_used -= (compensation / m_costPerBit);
                    // GLOOP_DATA_LOG("  boosting candidate[%u], polling:(%lld),ticks:(%lld), previous:(%lld)\n", m_id, (long long int)m_burned.count(), (long long int)used().count(), (long long int)value.count());
                }
            }

            m_kernelLock.lock();
            while (!m_server.isAllowed(*this)) {
                GLOOP_DEBUG("[%u] Sleep\n", m_id);
                m_server.condition().wait(m_kernelLock);
            }

            {
                std::lock_guard<Lock> guard(m_lock);
                GLOOP_DATA_LOG("[%u] Lock kernel token.\n", m_id);

                m_killed = false;
                m_timeWatch.begin();
                m_attemptToLaunch.store(false);
                configureTick(m_timer);
            }
        }
        return true;
    }

    case Command::Type::Unlock: {
        {
            // bool killed = false;
            {
                std::lock_guard<Lock> guard(m_lock);
                m_timer.cancel();
                m_killTimer.end();
                // killed = m_killed;
                m_timeWatch.end();

                m_burned = Duration(0);
                m_scheduledDuringIO.store(false);
                Command::ReleaseStatus status = static_cast<Command::ReleaseStatus>(command.payload);
                if (status == Command::ReleaseStatus::Ready) {
                    // This flag makes the current ready to schedule.
                    m_attemptToLaunch.store(true);
                }

                {
                    std::lock_guard<Lock> serverStatusGuard(m_server.serverStatusLock());
                    m_used += (m_timeWatch.ticks() * m_costPerBit);
                    m_server.calculateNextSession(serverStatusGuard);
                }
                m_kernelLock.unlock();
                // GLOOP_DATA_LOG("[%u] Unlock kernel token, used:(%llu).\n", m_id, (long long unsigned)m_used.count());
                m_server.condition().notify_all();
            }
            // if (killed) {
            //     printf("%u %lld\n", m_id, (long long int)m_killTimer.ticks().count());
            // }
        }
        return false;
    }

    case Command::Type::IO: {
        GLOOP_UNREACHABLE();
    }
    }
    return false;
}

bool Session::initialize(Command& command)
{
    m_requestQueue = createQueue(GLOOP_SHARED_REQUEST_QUEUE, id(), true);
    m_responseQueue = createQueue(GLOOP_SHARED_RESPONSE_QUEUE, id(), true);
    m_sharedMemory = createMemory(GLOOP_SHARED_MEMORY, id(), GLOOP_SHARED_MEMORY_SIZE, true);
    m_signal = make_unique<boost::interprocess::mapped_region>(*m_sharedMemory.get(), boost::interprocess::read_write, /* Offset. */ 0, GLOOP_SHARED_MEMORY_SIZE);

    assert(m_requestQueue);
    assert(m_responseQueue);

    // FIXME: This is not correct manner. Just for testing.
    uint64_t costPerBit = command.payload;
    if (costPerBit != 0)
        m_costPerBit = costPerBit;
    // GLOOP_DATA_LOG("[%u] costPerBit:(%u)\n", (unsigned)id(), (unsigned)costPerBit);

    // NOTE: This initialize method is always executed in the single event loop thread.
    m_server.registerSession(*this);
    m_thread = make_unique<boost::thread>(&Session::main, this);

    command = (Command) {
        .type = Command::Type::Initialize,
        .payload = id()
    };
    return true;
}

void Session::main()
{
    while (true) {
        unsigned int priority { };
        std::size_t size { };
        Command command { };
        if (m_requestQueue->try_receive(&command, sizeof(Command), size, priority)) {
            if (handle(command)) {
                m_responseQueue->send(&command, sizeof(Command), 0);
            }
        } else {
            // FIXME
            boost::this_thread::interruption_point();
        }
    }
}

void Session::burnUsed(const Duration& currentVirtualTime)
{
    m_used -= currentVirtualTime;
    if (m_used < Duration(0)) {
        m_burned += -m_used;
        m_used = Duration(0);
    }
}

void Session::setUsed(const Duration& used)
{
    m_used = used;
}

} }  // namsepace gloop::monitor
