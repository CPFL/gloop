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
#ifndef GLOOP_H_
#define GLOOP_H_
#include <type_traits>
namespace gloop {

struct open_vec {
    char* filename;
    int mode;
};

template<typename Callback>
__device__ auto open(char* filename, int mode, Callback callback) -> typename std::result_of<Callback(int)>::type
{
    int fd = gopen(filename, mode);
    return callback(fd);
}

template<typename Callback>
__device__ auto write(int fd, size_t offset, size_t count, unsigned char* buffer, Callback callback) -> typename std::result_of<Callback(size_t)>::type
{
    size_t written_size = gwrite(fd, offset, count, buffer);
    return callback(written_size);
}

template<typename Callback>
__device__ auto fstat(int fd, Callback callback) -> typename std::result_of<Callback(size_t)>::type
{
    size_t value = ::fstat(fd);
    return callback(value);
}

template<typename Callback>
__device__ auto close(int fd, Callback callback) -> typename std::result_of<Callback(int)>::type
{
    int err = gclose(fd);
    return callback(err);
}

template<typename Callback>
__device__ auto read(int fd, size_t offset, size_t size, unsigned char* buffer, Callback callback) -> typename std::result_of<Callback(size_t)>::type
{
    size_t bytes_read = gread(fd, offset, size, buffer);
    return callback(bytes_read);
}

}  // namespace gloop
#endif  // GLOOP_H_
