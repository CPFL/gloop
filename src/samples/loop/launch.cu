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

#include <gloop/benchmark.h>
#include <gloop/initialize.cuh>
#include <gloop/utility.cuh>

__global__ void throttle(int count, int limit)
{
    for (; count < limit; ++count) {
    }
}

int main(int argc, char** argv) {

    if(argc<4) {
        fprintf(stderr,"<kernel_iterations> <blocks> <threads>\n");
        return -1;
    }
    int trials=atoi(argv[1]);
    int nblocks=atoi(argv[2]);
    int nthreads=atoi(argv[3]);
    int id=atoi(argv[4]);

    fprintf(stderr," iterations: %d blocks %d threads %d id %d\n",trials, nblocks, nthreads, id);
    cudaStream_t pgraph;
    GLOOP_CUDA_SAFE_CALL(cudaStreamCreate(&pgraph));

    gloop::Benchmark benchmark;
    benchmark.begin();
    throttle<<<nblocks, nthreads, 0, pgraph>>>(0, trials);
    GLOOP_CUDA_SAFE_CALL(cudaGetLastError());
    GLOOP_CUDA_SAFE_CALL(cudaStreamSynchronize(pgraph));
    benchmark.end();
    printf("[%d] ", id);
    benchmark.report();

    return 0;
}
