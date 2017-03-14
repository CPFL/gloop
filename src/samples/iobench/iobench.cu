/*
  Copyright (C) 2017 Yusuke Suzuki <yusuke.suzuki@sslab.ics.keio.ac.jp>

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

#include <gloop/benchmark.h>
#include <gloop/statistics.h>
#include <gloop/gloop.h>

__device__ void gmain(gloop::DeviceLoop<>* loop, int fd, unsigned char* buffer, int count, int limit, int ioSize, int loopCount)
{
    volatile int res = 20;
    for (int i = 0; i < loopCount; ++i)
        res = i;

    if (count != limit) {
        gloop::fs::read(loop, fd, 0, ioSize, buffer, [=](gloop::DeviceLoop<>* loop, int) {
            gmain(loop, fd, buffer, count + 1, limit, ioSize, loopCount);
        });
        return;
    }
    gloop::fs::close(loop, fd, [=](gloop::DeviceLoop<>* loop, int error) { });
}

int main(int argc, char** argv)
{
    if (argc < 5) {
        fprintf(stderr, "<trials> <blocks> <pblocks> <threads> <id> <ioSize> <loopCount>\n");
        return -1;
    }
    int trials = atoi(argv[1]);
    int nblocks = atoi(argv[2]);
    int physblocks = atoi(argv[3]);
    int nthreads = atoi(argv[4]);
    int id = atoi(argv[5]);
    int ioSize = atoi(argv[6]);
    int loopCount = atoi(argv[7]);

    fprintf(stderr, " iterations: %d blocks %d threads %d id %d ioSize %d, loops %d\n", trials, nblocks, nthreads, id, ioSize, loopCount);

    {
        gloop::Statistics::Scope<gloop::Statistics::Type::GPUInit> scope;
        uint32_t pipelinePageCount = 0;
        dim3 blocks(nblocks);
        dim3 psblocks(physblocks);
        std::unique_ptr<gloop::HostLoop> hostLoop = gloop::HostLoop::create(0);
        std::unique_ptr<gloop::HostContext> hostContext = gloop::HostContext::create(*hostLoop, psblocks, pipelinePageCount);

        {
            std::lock_guard<gloop::HostLoop::KernelLock> lock(hostLoop->kernelLock());
            CUDA_SAFE_CALL(cudaDeviceSetLimit(cudaLimitMallocHeapSize, (1ULL << 20)));
        }

        {
            gloop::Statistics::Scope<gloop::Statistics::Type::Kernel> scope;
            hostLoop->launch(*hostContext, blocks, nthreads, [] GLOOP_DEVICE_LAMBDA(gloop::DeviceLoop<> * loop, int trials, int ioSize, int loopCount) {
                gloop::fs::open(loop, "tmp/dump", O_RDONLY, [=](gloop::DeviceLoop<>* loop, int fd) {
                    unsigned char* buffer = static_cast<unsigned char*>(malloc(ioSize));
                    gmain(loop, fd, buffer, 0, trials, ioSize, loopCount);
                });
            }, trials, ioSize, loopCount);
        }
    }
    gloop::Statistics::instance().report(stderr);

    return 0;
}
