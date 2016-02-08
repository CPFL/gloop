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
#ifndef GLOOP_IO_CU_H_
#define GLOOP_IO_CU_H_
#include <memory>
#include <unordered_map>
#include "noncopyable.h"
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

class FileDescriptorTable {
GLOOP_NONCOPYABLE(FileDescriptorTable)
public:
    FileDescriptorTable() { };
    ~FileDescriptorTable();

    int open(std::string fileName, int mode);
    void close(int fd);

    void mmap(void* host, void* device);
    void* munmap(void* device);

private:
    // This merges file open requests from the blocks.
    typedef std::unordered_map<std::string, std::shared_ptr<File>> FileNameToFileMap;
    FileNameToFileMap m_fileNameToFile;

    typedef std::unordered_map<void*, void*> MmapTable;
    MmapTable m_mmapTable;
};

}  // namespace gloop
#endif  // GLOOP_IO_CU_H_
