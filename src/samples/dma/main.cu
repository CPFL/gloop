/*
  Copyright (C) 2018 Yusuke Suzuki <yusuke.suzuki@sslab.ics.keio.ac.jp>

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

#include <cuda.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <math.h>
#include <assert.h>

#define MAX_TRIALS (10)
double time_res[MAX_TRIALS];

int global_devicenum;

template<typename Function>
float measure(cudaStream_t stream, Function function)
{
    float timeInMS = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream);
    function(stream);
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timeInMS, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamSynchronize(stream);
    return timeInMS;
}

int main( int argc, char** argv)
{

    char* gpudev=getenv("GPUDEVICE");
    global_devicenum=0;
    if (gpudev!=NULL) global_devicenum=atoi(gpudev);

    fprintf(stderr,"GPU device chosen %d\n",global_devicenum);

    if(argc<2) {
        fprintf(stderr,"<kernel_iterations>\n");
        return -1;
    }
    int trials=atoi(argv[1]);

    cudaFree(nullptr);

    void* devicePtr = nullptr;
    void* hostPtr = nullptr;
    size_t maxSize = (4096ULL << 20);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMalloc(&devicePtr, maxSize);
    cudaHostAlloc(&hostPtr, maxSize, cudaHostAllocDefault);

    double total_time=0;
    size_t total_size;

    memset(time_res,0,MAX_TRIALS*sizeof(double));

    for(int i=0;i<trials;i++) {
        for (int j = 0; j < 16 * 16 * 16; ++j) {
            size_t size = 4096ULL * j;
            {
                float time = measure(stream, [&] (cudaStream_t stream) {
                    cudaMemcpyAsync(hostPtr, devicePtr, size, cudaMemcpyDeviceToHost, stream);
                });
                fprintf(stderr, "DToH %llu %f\n", size, static_cast<double>(time));
            }
            {
                float time = measure(stream, [&] (cudaStream_t stream) {
                    cudaMemcpyAsync(devicePtr, hostPtr, size, cudaMemcpyHostToDevice, stream);
                });
                fprintf(stderr, "HToD %llu %f\n", size, static_cast<double>(time));
            }
            cudaStreamSynchronize(stream);
        }
    }
    return 0;
}



