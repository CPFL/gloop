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

#pragma once

#include "benchmark.h"
#include "command.h"
#include "config.h"
#include "monitor_lock.h"
#include "noncopyable.h"
#include <atomic>
#include <boost/asio.hpp>
#include <boost/asio/high_resolution_timer.hpp>
#include <boost/interprocess/ipc/message_queue.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/intrusive/list_hook.hpp>
#include <boost/thread.hpp>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
namespace gloop {
namespace monitor {

class Server;

class Session : public boost::intrusive::list_base_hook<> {
    GLOOP_NONCOPYABLE(Session);

public:
    typedef std::chrono::microseconds Duration;
    Session(Server&, uint32_t id);
    ~Session();

    static Duration boostThreshold()
    {
        return std::chrono::microseconds(GLOOP_ROUGH_TIMESLICE);
    }

    boost::asio::local::stream_protocol::socket& socket()
    {
        return m_socket;
    }
    uint32_t id() const
    {
        return m_id;
    }

    typedef std::aligned_storage<sizeof(Command), std::alignment_of<Command>::value>::type CommandBuffer;

    bool isAttemptingToLaunch() const
    {
        return m_attemptToLaunch.load();
    }

    bool isScheduledDuringIO() const
    {
        return m_scheduledDuringIO.load();
    }
    void setScheduledDuringIO()
    {
        m_scheduledDuringIO.store(true);
    }

    void handShake();

    const Duration& used() const
    {
        return m_used;
    }
    Duration& used()
    {
        return m_used;
    }

    void burnUsed(const Duration&);

    uint64_t costPerBit() const
    {
        return m_costPerBit;
    }

    int64_t readAndClearUtil()
    {
        int64_t us = m_util.count();
        m_util = Duration(0);
        return us;
    }

private:
    Command* buffer()
    {
        return reinterpret_cast<Command*>(&m_buffer);
    }

    void configureTick(boost::asio::high_resolution_timer& timer);

    void main();
    bool handle(Command&);
    void handleRead(const boost::system::error_code& error);
    void handleWrite(const boost::system::error_code& error);

    bool initialize(Command&);

    void kill();

    std::atomic<bool> m_attemptToLaunch{false};
    std::atomic<bool> m_scheduledDuringIO{false};
    uint32_t m_id;
    Server& m_server;
    Lock m_lock;
    boost::asio::local::stream_protocol::socket m_socket;
    CommandBuffer m_buffer;
    std::unique_ptr<boost::thread> m_thread;
    std::unique_ptr<boost::interprocess::message_queue> m_requestQueue;
    std::unique_ptr<boost::interprocess::message_queue> m_responseQueue;
    std::unique_ptr<boost::interprocess::shared_memory_object> m_sharedMemory;
    std::unique_ptr<boost::interprocess::mapped_region> m_signal;
    std::unique_lock<ServerLock> m_kernelLock;
    boost::asio::high_resolution_timer m_timer;

    // Scheduler members.
    TimeWatch m_timeWatch;
    Duration m_used{0};
    Duration m_burned{0};
    Duration m_util{0};
    uint64_t m_costPerBit{1};

    bool m_killed{false};
    TimeWatch m_killTimer;
};
}
} // namsepace gloop::monitor
