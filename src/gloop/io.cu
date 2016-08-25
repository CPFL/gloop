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
#include "io.cuh"
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <fcntl.h>
#include <mutex>
#include <unistd.h>
namespace gloop {

FileDescriptorTable::~FileDescriptorTable()
{
    std::lock_guard<Mutex> locker(m_mutex);

    for (auto& pair : m_fileNameToFile) {
        ::close(pair.second->fd);
    }
}

int FileDescriptorTable::open(std::string fileName, int mode)
{
    std::lock_guard<Mutex> locker(m_mutex);

    // TODO: Check mode.
    auto iter = m_fileNameToFile.find(fileName);
    if (iter != m_fileNameToFile.end()) {
        iter->second->refCount++;
        return iter->second->fd;
    }
    int fd = ::open(fileName.c_str(), mode, 644);
    m_fileNameToFile.insert(iter, std::make_pair(fileName, std::make_shared<File>(fd)));
    return fd;
}

void FileDescriptorTable::close(int fd)
{
    std::lock_guard<Mutex> locker(m_mutex);

    auto iter = std::find_if(m_fileNameToFile.begin(), m_fileNameToFile.end(), [&](const FileNameToFileMap::value_type& pair) {
        return pair.second->fd == fd;
    });
    if (iter == m_fileNameToFile.end())
        return;
    if (--iter->second->refCount == 0) {
        ::close(iter->second->fd);
        m_fileNameToFile.erase(iter);

        // Invalidate opened mappings. It leads leaks. So you should unmap these mappings before closing the fd.
        std::vector<MmapRequest> vec;
        for (const auto& pair : m_mmapRequestsTable) {
            if (std::get<0>(pair.first) == fd)
                vec.push_back(pair.first);
        }
        for (auto tuple : vec) {
            auto iterator = m_mmapRequestsTable.find(tuple);
            std::shared_ptr<MmapResult> result = iterator->second;
            dropMmapResult(locker, result);
        }
    }
}

void FileDescriptorTable::registerMapping(void* device, std::shared_ptr<MmapResult> result)
{
    std::lock_guard<Mutex> locker(m_mutex);
    m_mmapTable.insert(std::make_pair(device, result));
}

void* FileDescriptorTable::lookupHostByDevice(void* device)
{
    std::lock_guard<Mutex> locker(m_mutex);
    return m_mmapTable[device]->host;
}

std::shared_ptr<MmapResult> FileDescriptorTable::unregisterMapping(void* device)
{
    std::lock_guard<Mutex> locker(m_mutex);
    auto iterator = m_mmapTable.find(device);
    if (iterator == m_mmapTable.end())
        return nullptr;
    std::shared_ptr<MmapResult> result = iterator->second;
    result->refCount--;
    return result;
}

bool FileDescriptorTable::requestMmap(int fd, size_t offset, size_t size, std::shared_ptr<MmapResult>& result)
{
    std::lock_guard<Mutex> locker(m_mutex);
    MmapRequest request = std::make_tuple(fd, offset, size);
    auto iterator = m_mmapRequestsTable.find(request);
    if (iterator == m_mmapRequestsTable.end()) {
        result = std::make_shared<MmapResult>(request);
        m_mmapRequestsTable.insert(iterator, std::make_pair(request, result));
        return true;
    }
    result = iterator->second;
    result->refCount++;
    return false;
}

void FileDescriptorTable::dropMmapResult(const std::lock_guard<Mutex>&, std::shared_ptr<MmapResult> result)
{
    assert(result->refCount == 0);
    m_mmapTable.erase(result->device);
    m_mmapRequestsTable.erase(result->request);
}

void FileDescriptorTable::dropMmapResult(std::shared_ptr<MmapResult> result)
{
    std::lock_guard<Mutex> locker(m_mutex);
    dropMmapResult(locker, result);
}

} // namespace gloop
