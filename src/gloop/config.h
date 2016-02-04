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
#ifndef GLOOP_CONFIG_H_
#define GLOOP_CONFIG_H_

#define GLOOP_VERSION "0.0.1"
#define GLOOP_ENDPOINT "/tmp/gloop_endpoint"
#define GLOOP_SHARED_MAIN_QUEUE "gloop_shared_main_queue_"
#define GLOOP_SHARED_REQUEST_QUEUE "gloop_shared_request_queue_"
#define GLOOP_SHARED_RESPONSE_QUEUE "gloop_shared_response_queue_"
#define GLOOP_SHARED_MEMORY "gloop_shared_memory_"
#define GLOOP_SHARED_MEMORY_SIZE 0x1000UL
#define GLOOP_KILL_TIME 10

#define GLOOP_SHARED_SLOT_SIZE 64
#define GLOOP_SHARED_PAGE_SIZE 4096UL
#define GLOOP_SHARED_PAGE_COUNT 2
#define GLOOP_THREAD_GROUP_SIZE 8

namespace gloop {
}  // namespace gloop
#endif  // GLOOP_CONFIG_H_
