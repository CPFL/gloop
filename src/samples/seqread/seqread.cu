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
#include <gloop/device_memory.cuh>


__device__ void perform_read(gloop::DeviceLoop* loop, uchar* scratch, int fd, size_t me, size_t filesize)
{
    if (me < filesize) {
        size_t toRead = min((size_t)GLOOP_SHARED_PAGE_SIZE, (size_t)(filesize - me));
        gloop::fs::read(loop, fd, me, toRead, scratch, [=](gloop::DeviceLoop* loop, int read) {
            if (toRead != read) {
                assert(NULL);
            }

            perform_read(loop, scratch, fd, me + GLOOP_SHARED_PAGE_SIZE * gloop::logicalGridDim.x, filesize);
        });
        return;
    }

    gloop::fs::close(loop, fd, [=](gloop::DeviceLoop* loop, int err) {
    });
}

__device__ void entry(gloop::DeviceLoop* loop, char* filename)
{
    __shared__ uchar* scratch;

    BEGIN_SINGLE_THREAD
    {
        scratch=(uchar*)malloc(GLOOP_SHARED_PAGE_SIZE);
        GLOOP_ASSERT(scratch!=NULL);
    }
    END_SINGLE_THREAD

    gloop::fs::open(loop, filename, O_RDONLY, [=](gloop::DeviceLoop* loop, int fd) {
        gloop::fs::fstat(loop, fd, [=](gloop::DeviceLoop* loop, int filesize) {
            size_t me = gloop::logicalBlockIdx.x * GLOOP_SHARED_PAGE_SIZE;
            perform_read(loop, scratch, fd, me, filesize);
        });
    });
}

int main(int argc, char** argv) {

    if(argc<5) {
        fprintf(stderr,"<kernel_iterations> <blocks> <threads> file\n");
        return -1;
    }
    int trials = atoi(argv[1]);
    int nblocks = atoi(argv[2]);
    int nthreads = atoi(argv[3]);

    fprintf(stderr," iterations: %d blocks %d threads %d\n",trials, nblocks, nthreads);

    {
        dim3 blocks(nblocks);
        std::unique_ptr<gloop::HostLoop> hostLoop = gloop::HostLoop::create(0);
        std::unique_ptr<gloop::HostContext> hostContext = gloop::HostContext::create(*hostLoop, blocks);
        const std::string filename(argv[4]);

        std::shared_ptr<gloop::DeviceMemory> memory = gloop::DeviceMemory::create(filename.size() + 1);
        CUDA_SAFE_CALL(cudaMemcpy(memory->devicePointer(), filename.c_str(), filename.size() + 1,cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaDeviceSetLimit(cudaLimitMallocHeapSize, (1ULL << 20) * 896));

        gloop::Benchmark bench;
        bench.begin();
        hostLoop->launch(*hostContext, nthreads, [=] GLOOP_DEVICE_LAMBDA (gloop::DeviceLoop* loop, char* filename) {
            entry(loop, filename);
        }, reinterpret_cast<char*>(memory->devicePointer()));
        bench.end();
        bench.report();
    }

    return 0;
}
