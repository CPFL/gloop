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

#include "hash_tuple.h"
#include "noncopyable.h"
#include "spinlock.h"
#include <memory>
#include <mutex>
#include <tuple>
#include <unordered_map>
namespace gloop {

struct File {
    File(int fd);

    int fd;
    int refCount;
};

inline File::File(int fd)
    : fd(fd)
    , refCount(1)
{
}

// fd, offset, size
typedef std::tuple<int, size_t, size_t> MmapRequest;

struct MmapResult {
    MmapResult(MmapRequest);

    void* host{nullptr};
    void* device{nullptr};
    size_t size{0};
    int refCount{1};
    MmapRequest request;
};

inline MmapResult::MmapResult(MmapRequest request)
    : request(request)
{
}

class FileDescriptorTable {
    GLOOP_NONCOPYABLE(FileDescriptorTable)
public:
    typedef Spinlock Mutex;

    FileDescriptorTable(){};
    ~FileDescriptorTable();

    int open(std::string fileName, int mode);
    void close(int fd);

    void registerMapping(void* device, std::shared_ptr<MmapResult> result);
    void* lookupHostByDevice(void* device);
    std::shared_ptr<MmapResult> unregisterMapping(void* device);

    bool requestMmap(int fd, size_t offset, size_t size, std::shared_ptr<MmapResult>& result);
    void dropMmapResult(std::shared_ptr<MmapResult> result);
    void dropMmapResult(const std::lock_guard<Mutex>&, std::shared_ptr<MmapResult> result);

private:
    // This merges file open requests from the blocks.
    typedef std::unordered_map<std::string, std::shared_ptr<File>> FileNameToFileMap;
    FileNameToFileMap m_fileNameToFile;

    typedef std::unordered_map<void*, std::shared_ptr<MmapResult>> MmapTable;
    MmapTable m_mmapTable;

    typedef std::unordered_map<MmapRequest, std::shared_ptr<MmapResult>, hash_tuple::hash<MmapRequest>> MmapRequests;
    MmapRequests m_mmapRequestsTable;

    Mutex m_mutex{};
};

} // namespace gloop
