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

#include <iostream>

#define checkCudaErrors(expr) expr

template<typename Callback>
__global__ void axpy(float a, float* x, float* y, Callback callback) {
    callback(a, x, y);
    // y[threadIdx.x] = a * x[threadIdx.x];
}

int main(int argc, char* argv[]) {
    const int kDataLen = 4;

    float a = 2.0f;
    float host_x[kDataLen] = {1.0f, 2.0f, 3.0f, 4.0f};
    float host_y[kDataLen];

    // Copy input data to device.
    float* device_x;
    float* device_y;
    checkCudaErrors(cudaMalloc(&device_x, kDataLen * sizeof(float)));
    checkCudaErrors(cudaMalloc(&device_y, kDataLen * sizeof(float)));
    checkCudaErrors(cudaMemcpy(device_x, host_x, kDataLen * sizeof(float), cudaMemcpyHostToDevice));

    // Launch the kernel.
    // axpy<<<1, kDataLen>>>(a, device_x, device_y);
    axpy<<<1, kDataLen>>>(a, device_x, device_y, [=] (float a, float* x, float* y) {
        y[threadIdx.x] = a * x[threadIdx.x];
    });

    // Copy output data to host.
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(host_y, device_y, kDataLen * sizeof(float), cudaMemcpyDeviceToHost));

    // Print the results.
    for (int i = 0; i < kDataLen; ++i) {
        std::cout << "y[" << i << "] = " << host_y[i] << "\n";
    }

    checkCudaErrors(cudaDeviceReset());
    return 0;
}
