/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/* Matrix multiplication: C = A * B.
 * Device code.
 */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>
#include <sys/mman.h>
#include <gloop/gloop.h>

#define CHECK_BANK_CONFLICTS 0
#if CHECK_BANK_CONFLICTS
#define AS(i, j) cutilBankChecker(((float*)&As[0][0]), (BLOCK_SIZE * i + j))
#define BS(i, j) cutilBankChecker(((float*)&Bs[0][0]), (BLOCK_SIZE * i + j))
#else
#define AS(i, j) As[i][j]
//#define AS(i, j) *((&As[0][0])+(i)*(BLOCK_SIZE+1)+(j))
#define BS(i, j) *((&Bs[0][0])+(i)*(BLOCK_SIZE+1)+(j))
#define CS(i, j) Cs[i][j]
#endif


#include "fs_globals.cu.h"
#include "fs_calls.cu.h"

__device__ volatile INIT_LOCK init_lock;
__device__ volatile LAST_SEMAPHORE last_lock;

////////////////////////////////////////////////////////////////////////////////
//! Matrix multiplication on the device: C = A * B
//! wA is A's width and wB is B's width
////////////////////////////////////////////////////////////////////////////////

template <int BLOCK_SIZE> __global__ void
matrixMulCUDA(float *C, float *A, float *B, int wA, int wB)
{
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
        //#pragma unroll

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Csub += As[ty][k] * Bs[k][tx];
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}

__device__ float tmp_a[1<<22];
__device__ float tmp_b[1<<22];
__device__ float tmp_c[1<<22];

template <int BLOCK_SIZE> __device__ void
performOuterLoop(gloop::DeviceLoop* loop, int wA, int wB, int perBlockX, int perBlockY, int f_a, int f_b, int f_c, int by);

template <int BLOCK_SIZE> __device__ void
performInnerLoop(gloop::DeviceLoop* loop, int wA, int wB, int perBlockX, int perBlockY, int f_a, int f_b, int f_c, volatile float* ptr_a, volatile float* ptr_c, int by, int bx)
{
    if (bx == perBlockX) {
        auto nextProcess = [=](gloop::DeviceLoop* loop, int error) {
            gloop::fs::munmap(loop, ptr_c, wA*BLOCK_SIZE*sizeof(float), [=](gloop::DeviceLoop* loop, int error) {
                gloop::fs::munmap(loop, ptr_a, wA*BLOCK_SIZE*sizeof(float), [=](gloop::DeviceLoop* loop, int error) {
                    performOuterLoop<BLOCK_SIZE>(loop, wA, wB, perBlockX, perBlockY, f_a, f_b, f_c, by + 1);
                });
            });
        };
        if(perBlockX>4) {
            gloop::fs::msync(loop, ptr_c, wA*BLOCK_SIZE*sizeof(float), 0, nextProcess);
            return;
        }
        nextProcess(loop, 0);
        return;
    }

    // Index of the first sub-matrix of A processed by the block
    int hB=wA;
    int bBegin = hB*BLOCK_SIZE*bx*sizeof(float);
    gloop::fs::mmap(loop, NULL,wA*BLOCK_SIZE*sizeof(float), PROT_READ | PROT_WRITE, MAP_SHARED, f_b, bBegin, [=](gloop::DeviceLoop* loop, volatile void* res) {
        volatile float* ptr_b = (volatile float*)res;
        //        volatile float * ptr_b=tmp_b;
        //        if (ptr_b==GMAP_FAILED) GLOOP_ERROR("GMMAP failed with m_b");

        // Block index

        // Thread index
        int tx = threadIdx.x;
        int ty = threadIdx.y;


        // Index of the last sub-matrix of A processed by the block
        int aEnd   =  wA - 1;

        // Step size used to iterate through the sub-matrices of A
        int aStep  = BLOCK_SIZE;

        // Csub is used to store the element of the block sub-matrix
        // that is computed by the thread
        float Csub = 0;

        for( int a = 0, b= 0; a <=aEnd; a += aStep, b += aStep) {

            // Declaration of the shared memory array As used to
            // store the sub-matrix of A
            __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

            // Declaration of the shared memory array Bs used to
            // store the sub-matrix of B
            __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE+1];


            // Load the matrices from device memory
            // to shared memory; each thread loads
            // one element of each matrix

            AS(ty, tx) = ptr_a[a + wA * ty + tx];
            BS(ty, tx) = ptr_b[b + wA * ty + tx];
            //   AS(ty,tx)=1;
            //   BS(ty,tx)=1;

            // Synchronize to make sure the matrices are loaded
            __syncthreads();

            // Multiply the two matrices together;
            // each thread computes one element
            // of the block sub-matrix
            //  #pragma unroll
            float* bs_ptr=&BS(tx,0);
            for (int k = 0; k < BLOCK_SIZE; ++k){
                Csub+=AS(ty,k)*(*(bs_ptr+k));;
            }
            //if (Csub!=-1) Csub=AS(0,0);
            // Synchronize to make sure that the preceding
            // computation is done before loading two new
            // sub-matrices of A and B in the next iteration
            __syncthreads();
        }

        ptr_c[bx*BLOCK_SIZE+ wB*ty+tx]=Csub;

        //    }
        // Write the block sub-matrix to device memory;
        // each thread writes one element

        gloop::fs::munmap(loop, ptr_b, wA*BLOCK_SIZE*sizeof(float), [=](gloop::DeviceLoop* loop, int error) {
            performInnerLoop<BLOCK_SIZE>(loop, wA, wB, perBlockX, perBlockY, f_a, f_b, f_c, ptr_a, ptr_c, by, bx + 1);
        });
    });
}

template <int BLOCK_SIZE> __device__ void
performOuterLoop(gloop::DeviceLoop* loop, int wA, int wB, int perBlockX, int perBlockY, int f_a, int f_b, int f_c, int by)
{
    if (by == (blockIdx.y+1)*perBlockY) {
        gloop::fs::close(loop, f_a, [=](gloop::DeviceLoop* loop, int error) {
            gloop::fs::close(loop, f_b, [=](gloop::DeviceLoop* loop, int error) {
                gloop::fs::close(loop, f_c, [=](gloop::DeviceLoop* loop, int error) {
                });
            });
        });
        return;
    }

    int wC=wB;
    int cBegin = wC*BLOCK_SIZE*by*sizeof(float);

    gloop::fs::mmap(loop, NULL,wA*BLOCK_SIZE*sizeof(float),PROT_READ | PROT_WRITE, MAP_SHARED, f_c,cBegin, [=](gloop::DeviceLoop* loop, volatile void* res) {
        volatile float* ptr_c=(volatile float*)res;
        //    volatile float * ptr_c=tmp_c;
        if (ptr_c==GMAP_FAILED) GLOOP_ERROR("GMMAP failed with m_c");

        int aBegin = wA*BLOCK_SIZE*by*sizeof(float);

        gloop::fs::mmap(loop, NULL,wA*BLOCK_SIZE*sizeof(float),PROT_READ | PROT_WRITE, MAP_SHARED, f_a,aBegin, [=](gloop::DeviceLoop* loop, volatile void* res) {
            volatile float* ptr_a=(volatile float*)res;
            //      volatile float* ptr_a=tmp_a;
            if (ptr_a==GMAP_FAILED) GLOOP_ERROR("GMMAP failed with m_a");

#if 0
            gloop::fs::munmap(loop, ptr_c, wA*BLOCK_SIZE*sizeof(float), [=](gloop::DeviceLoop* loop, int error) {
                gloop::fs::munmap(loop, ptr_a, wA*BLOCK_SIZE*sizeof(float), [=](gloop::DeviceLoop* loop, int error) {
                });
            });
#endif

            int bx = 0;
            performInnerLoop<BLOCK_SIZE>(loop, wA, wB, perBlockX, perBlockY, f_a, f_b, f_c, ptr_a, ptr_c, by, bx);
        });
    });
}

template <int BLOCK_SIZE> __device__ void
performOperation(gloop::DeviceLoop* loop, int wA, int wB, int perBlockX, int perBlockY, int f_a, int f_b, int f_c)
{
    // for (int by=blockIdx.y*perBlockY;by<(blockIdx.y+1)*perBlockY;by++){
    int by=blockIdx.y*perBlockY;
    performOuterLoop<BLOCK_SIZE>(loop, wA, wB, perBlockX, perBlockY, f_a, f_b, f_c, by);
}

template <int BLOCK_SIZE> __device__ void
matrixMul(gloop::DeviceLoop* loop, int wA, int wB, int perBlockX, int perBlockY, char n)
{
    gloop::fs::open(loop, "mtx_a", O_RDWR, [=](gloop::DeviceLoop* loop, int f_a) {
        if (f_a<0) {
            GLOOP_ERROR("Failed to open a");
        }

        gloop::fs::open(loop, "mtx_b", O_RDWR, [=](gloop::DeviceLoop* loop, int f_b) {
            if (f_b<0) {
                GLOOP_ERROR("Failed to open B");
            }

            char out[6]="mtx_c"; out[0]=n;
            gloop::fs::open(loop, out, O_RDWR, [=](gloop::DeviceLoop* loop, int f_c) {
                if (f_c<0) {
                    GLOOP_ERROR("Failed to open c");
                }
                performOperation<BLOCK_SIZE>(loop, wA, wB, perBlockX, perBlockY, f_a, f_b, f_c);
            });
        });
    });
}


void init_app()
{
    // INITI LOCK
    void* inited;

    CUDA_SAFE_CALL(cudaGetSymbolAddress(&inited,init_lock));
    CUDA_SAFE_CALL(cudaMemset(inited,0,sizeof(INIT_LOCK)));

    CUDA_SAFE_CALL(cudaGetSymbolAddress(&inited,last_lock));
    CUDA_SAFE_CALL(cudaMemset(inited,0,sizeof(LAST_SEMAPHORE)));
}

#endif // #ifndef _MATRIXMUL_KERNEL_H_
