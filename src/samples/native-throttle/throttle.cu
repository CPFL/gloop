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

#include <cstdint>
#include <cstdio>
#include <gloop/benchmark.h>

__global__ void throttle(int count, int limit)
{
}

int main(int argc, char** argv) {

    if(argc<5) {
        fprintf(stderr,"<kernel_iterations> <blocks> <pblocks> <threads>\n");
        return -1;
    }
    int trials=atoi(argv[1]);
    int nblocks=atoi(argv[2]);
    int physblocks=atoi(argv[3]);
    int nthreads=atoi(argv[4]);
    int id=atoi(argv[5]);

    fprintf(stderr," iterations: %d blocks %d threads %d id %d\n",trials, nblocks, nthreads, id);

    {
        uint32_t pipelinePageCount = 0;
        dim3 blocks(nblocks);
        dim3 psblocks(physblocks);
        // cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 1 << 30);
        cudaDeviceSynchronize();
        void* devicePointer = nullptr;
        cudaMalloc(&devicePointer, 16);
        gloop::Benchmark benchmark;
        benchmark.begin();
        for (int i = 0; i < trials; ++i) {
            throttle<<<blocks, 1>>>(i, trials);
        }
        cudaDeviceSynchronize();
        benchmark.end();
        printf("[%d] ", id);
        benchmark.report();
    }

    return 0;
}
