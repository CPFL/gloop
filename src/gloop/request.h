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

#include <netinet/in.h>
#include <sys/socket.h>
#include "net_socket.h"
namespace gloop {
namespace request {

// Derived from GPUfs.
#define GLOOP_FILENAME_SIZE 32

struct Filename {
    char data[GLOOP_FILENAME_SIZE];
};

struct Open {
    Filename filename;
    int mode;
};

struct OpenResult {
    int fd;
};

struct Write {
    int fd;
    size_t offset;
    size_t count;
    unsigned char* buffer;
};

struct WriteResult {
    ssize_t writtenCount;
};

struct Fstat {
    int fd;
};

struct FstatResult {
    off_t size;
};

struct Close {
    int fd;
};

struct CloseResult {
    int error;
};

struct Ftruncate {
    int fd;
    off_t offset;
};

struct FtruncateResult {
    int error;
};

struct Read {
    int fd;
    size_t offset;
    size_t count;
    unsigned char* buffer;
};

struct ReadResult {
    ssize_t readCount;
};

struct AllocOnePageResult {
    void* page;
};

struct Mmap {
    void* address;
    size_t size;
    int prot;
    int flags;
    int fd;
    off_t offset;
};

struct MmapResult {
    volatile void* address;
};

struct Munmap {
    volatile void* address;
    size_t size;
};

struct MunmapResult {
    int error;
};

struct Msync {
    volatile void* address;
    size_t size;
    int flags;
};

struct MsyncResult {
    int error;
};

struct NetTCPConnect {
    struct sockaddr_in address;
};

struct NetTCPConnectResult {
    net::Socket* socket;
};

struct NetTCPBind {
    struct sockaddr_in address;
};

struct NetTCPBindResult {
    net::Server* server;
};

struct NetTCPUnbind {
    net::Server* server;
};

struct NetTCPUnbindResult {
    int error;
};

struct NetTCPAccept {
    net::Server* server;
};

struct NetTCPAcceptResult {
    net::Socket* socket;
};

struct NetTCPReceive {
    net::Socket* socket;
    size_t count;
    unsigned char* buffer;
    int flags;
};

struct NetTCPReceiveResult {
    ssize_t receiveCount;
};

struct NetTCPSend {
    net::Socket* socket;
    size_t count;
    unsigned char* buffer;
};

struct NetTCPSendResult {
    ssize_t sentCount;
};

struct NetTCPClose {
    net::Socket* socket;
};

struct NetTCPCloseResult {
    int error;
};

struct Payload {
    union {
        Open open;
        OpenResult openResult;

        Write write;
        WriteResult writeResult;

        Fstat fstat;
        FstatResult fstatResult;

        Close close;
        CloseResult closeResult;

        Ftruncate ftruncate;
        FtruncateResult ftruncateResult;

        Read read;
        ReadResult readResult;

        AllocOnePageResult allocOnePageResult;

        Mmap mmap;
        MmapResult mmapResult;

        Munmap munmap;
        MunmapResult munmapResult;

        Msync msync;
        MsyncResult msyncResult;

        NetTCPConnect netTCPConnect;
        NetTCPConnectResult netTCPConnectResult;

        NetTCPBind netTCPBind;
        NetTCPBindResult netTCPBindResult;

        NetTCPUnbind netTCPUnbind;
        NetTCPUnbindResult netTCPUnbindResult;

        NetTCPAccept netTCPAccept;
        NetTCPAcceptResult netTCPAcceptResult;

        NetTCPReceive netTCPReceive;
        NetTCPReceiveResult netTCPReceiveResult;

        NetTCPSend netTCPSend;
        NetTCPSendResult netTCPSendResult;

        NetTCPClose netTCPClose;
        NetTCPCloseResult netTCPCloseResult;
    } u;
};

typedef Payload Request;

} }  // namespace gloop::request
