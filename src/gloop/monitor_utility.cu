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

#include "command.h"
#include "make_unique.h"
#include "monitor_utility.h"
namespace gloop {
namespace monitor {

std::string createName(const std::string& prefix, uint32_t id)
{
    std::vector<char> name(prefix.size() + 100);
    const int ret = std::snprintf(name.data(), name.size() - 1, "%s%u", prefix.c_str(), id);
    if (ret < 0) {
        std::perror(nullptr);
        std::exit(1);
    }
    name[ret] = '\0';
    return std::string(name.data(), ret);
}

std::unique_ptr<boost::interprocess::message_queue> createQueue(const std::string& prefix, uint32_t id, bool create)
{
    const std::string name = createName(prefix, id);
    if (create) {
        boost::interprocess::message_queue::remove(name.c_str());
        return make_unique<boost::interprocess::message_queue>(boost::interprocess::create_only, name.c_str(), 0x1000, sizeof(Command));
    }
    return make_unique<boost::interprocess::message_queue>(boost::interprocess::open_only, name.c_str());
}

std::unique_ptr<boost::interprocess::shared_memory_object> createMemory(const std::string& prefix, uint32_t id, std::size_t sharedMemorySize, bool create)
{
    const std::string name = createName(prefix, id);
    std::unique_ptr<boost::interprocess::shared_memory_object> memory;
    if (create) {
        boost::interprocess::shared_memory_object::remove(name.c_str());
        memory = make_unique<boost::interprocess::shared_memory_object>(boost::interprocess::create_only, name.c_str(), boost::interprocess::read_write);
    } else {
        memory = make_unique<boost::interprocess::shared_memory_object>(boost::interprocess::open_only, name.c_str(), boost::interprocess::read_write);
    }
    memory->truncate(sharedMemorySize);
    return memory;
}
}
} // namsepace gloop::monitor
