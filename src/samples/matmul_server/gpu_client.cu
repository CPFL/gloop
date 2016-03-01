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

#include <gloop/gloop.h>
#include <gloop/benchmark.h>
#include "microbench_util.h"

#define THREADS_PER_TB 256

__device__ void gpuMain(gloop::DeviceLoop* loop, struct sockaddr_in *addr) {
#if 0
	__shared__ int sock;
	int ret;
	sock = gconnect_in(addr);
	if (sock < 0) {
		BEGIN_SINGLE_THREAD_PART {
			gprintf4_single("ERROR: gconnect_in sock: %d", sock, 0, 0, 0);
		} END_SINGLE_THREAD_PART;
		return;
	}


	if (ret = gbench_send_recv_bw<BUF_SIZE, NR_MSG>(sock)) {
		printf("gbench_send_recv_bw ret: %d\n", ret);
		goto out;
	}

out:
	BEGIN_SINGLE_THREAD_PART {
		single_thread_gclose(sock);
	} END_SINGLE_THREAD_PART;
#endif
}

int main(int argc, char** argv)
{
    dim3 blocks(1);
    std::unique_ptr<gloop::HostLoop> hostLoop = gloop::HostLoop::create(0);
    std::unique_ptr<gloop::HostContext> hostContext = gloop::HostContext::create(*hostLoop, blocks);

    struct sockaddr* addr;
    struct sockaddr* dev_addr;
    {
        if (argc > 2) {
            std::lock_guard<gloop::HostLoop::KernelLock> lock(hostLoop->kernelLock());
            CUDA_SAFE_CALL(cudaDeviceSetLimit(cudaLimitMallocHeapSize, (2 << 20) * 256));
            gpunet_client_init(&addr, &dev_addr, argv[1], argv[2]);
        } else {
            gpunet_usage_client(argc, argv);
            exit(1);
        }
    }

    gloop::Benchmark benchmark;
    benchmark.begin();
    {
        hostLoop->launch(*hostContext, THREADS_PER_TB, [=] GLOOP_DEVICE_LAMBDA (gloop::DeviceLoop* loop, thrust::tuple<struct sockaddr*> tuple) {
            struct sockaddr* address;
            thrust::tie(address) = tuple;
            gpuMain(loop, (struct sockaddr_in*)address);
        }, dev_addr);
    }
    benchmark.end();
    printf("[%d] ", 0);
    benchmark.report();

    return 0;
}
