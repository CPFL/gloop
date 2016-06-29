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
/*
* This expermental software is provided AS IS.
* Feel free to use/modify/distribute,
* If used, please retain this disclaimer and cite
* "GPUfs: Integrating a file system with GPUs",
* M Silberstein,B Ford,I Keidar,E Witchel
* ASPLOS13, March 2013, Houston,USA
*/
/***
* Matrix product from files. This GPUfs example uses the original matmul from CUDA SDK
* but instead of reading data from memory it reads/writes it from/to files
*/



/* Matrix multiplication: C = A * B.
 * Host code.
 *
 * This sample implements matrix multiplication as described in Chapter 3
 * of the programming guide.
 * It has been written for clarity of exposition to illustrate various CUDA
 * programming principles, not with the goal of providing the most
 * performant generic kernel for matrix multiplication.
 *
 * CUBLAS provides high-performance matrix multiplication.
 * See also:
 * V. Volkov and J. Demmel, "Benchmarking GPUs to tune dense linear algebra,"
 * in Proc. 2008 ACM/IEEE Conf. on Superconducting (SC '08),
 * Piscataway, NJ: IEEE Press, 2008, pp. Art. 31:1-11.
 *
 */

// Utilities and system includes
#include <cmath>
#include <errno.h>
#include <cublas_v2.h>

#if 0
#include <sdkHelper.h>  // helper for shared functions common to CUDA SDK samples
#include <shrQATest.h>
#include <shrUtils.h>
#else
// CUDA 7.5 helper functions.
#include <helper_functions.h>
#include <helper_cuda.h>
#include "shrQATest.h"
#endif

#include <cuda_runtime.h>
#include <unistd.h>

#include <gloop/gloop.h>

#include "fs_debug.cu.h"
#include "fs_initializer.cu.h"
#include "host_loop.h"

#define shrLog printf
static char *sSDKsample = "matrixMul";

#include "matrixMul.h"

#include <stdio.h>
#include <sys/mman.h>

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

////////////////////////////////////////////////////////////////////////////////
//! Matrix multiplication on the device: C = A * B
//! wA is A's width and wB is B's width
////////////////////////////////////////////////////////////////////////////////

template <int BLOCK_SIZE> __global__ void
matrixMulCUDA(float *C, float *A, float *B, int wA, int wB)
{
    // Block index
    int bx = loop->logicalBlockIdx().x;
    int by = loop->logicalBlockIdx().y;

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

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest(int argc, char** argv);
void randomInit(float*, int);
void transpose(float*, float*, int,int);
void printDiff(float*, float*, int, int, int, float);

extern "C"
void computeGold(float*, const float*, const float*, unsigned int, unsigned int, unsigned int);

void inline checkError(cublasStatus_t status, const char* msg)
{
    if(status != CUBLAS_STATUS_SUCCESS){
        printf(msg);
        exit(-1);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
    shrQAStart(argc, argv);
  printf("[ %s ]\n", sSDKsample);
    //shrSetLogFileName ("matrixMul.txt");
    shrLog("%s\n\tStarting (CUDA and CUBLAS tests)...\n\n", argv[0]);

    runTest(argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////

int  write_mtx(const char* name, void* data, size_t size,size_t offset=0, int doclose=1, int fd=0){
    if(offset==0)
        unlink(name);

    if(offset==0)
        fd=open(name,O_WRONLY|O_CREAT,S_IRWXU);
    if (fd<0) {
        perror("cant open mtx\n");
        exit(-1);
    }
    if(pwrite(fd, data,size,offset)!=size) {
        perror("cant write\n");
        exit(-1);
    }
    //fsync(fd);

    if (doclose) {
        close(fd);
        fd=0;
    }
    return fd;
}
#define START_CLOCK(var) {(var)=_timestamp();}
#define STOP_CLOCK(var) {(var)=_timestamp()-(var);}
#define CLOCK_IT(var,accumulator, proc) START_CLOCK(var); {proc;} STOP_CLOCK(var); accumulator+=var;

bool runCUDA=true;
void runTest(int argc, char** argv)
{
    // use a larger block size for Fermi and above
    int block_size = 32;
    // Optional Command-line multiplier for matrix sizes
    unsigned int uiWA, uiHA, uiWB, uiHB, uiWC, uiHC;



    uiWA = WA;
    uiHA = HA;
    uiWB = WB;

    char* c_HA=getenv("HA");
    char* c_WA=getenv("WA");
    char* c_WB=getenv("WB");
    if (c_HA)
        uiHA=HA*atoi(c_HA);
    if (c_WA)
        uiWA=WA*atoi(c_WA);
    if (c_WB)
        uiWB=WB*atoi(c_WB);

    uiHB = uiWA;
    uiWC = uiWB;
    uiHC = uiHA;
    shrLog("\nUsing Matrix Sizes: A(%u x %u), B(%u x %u), C(%u x %u)\n\n",
    uiHA, uiWA, uiHB, uiWB, uiHC, uiWC);





    if(checkCmdLineFlag(argc, (const char**)argv, "device")) {
        int devID = getCmdLineArgumentInt(argc, (const char **)argv, "device=");
        if (devID < 0) {
            printf("Invalid command line parameters\n");
            exit(-1);
        } else {
            devID = gpuDeviceInit(devID);
            if (devID < 0) {
                printf("exiting...\n");
                shrQAFinishExit(argc, (const char **)argv, QA_FAILED);
                exit(-1);
            }
        }
    } else {
        checkCudaErrors( cudaSetDevice(gpuGetMaxGflopsDeviceId()) );
    }

    int devID;
    cudaDeviceProp props;

    // get number of SMs on this GPU
    checkCudaErrors(cudaGetDevice(&devID));
    checkCudaErrors(cudaGetDeviceProperties(&props, devID));

    cudaSetDevice(0);
    printf("OK?\n");

    // setup execution parameters
    dim3 threads(block_size, block_size);
    //   int perBlockX=1;
    //   int perBlockY=1;
    int NUM_BLOCKS=104;
    if (uiHC<104*32)
        NUM_BLOCKS=uiHC/32;
    int perBlockX=uiWC / threads.x/1;
    int perBlockY=uiHC / threads.y/NUM_BLOCKS;

    dim3 grid( 1, NUM_BLOCKS);
    dim3 gridCUDA(grid.x*perBlockX,grid.y*perBlockY);
    printf(" grid size: %dx%d per blockX= %d per blockY= %d\n",grid.x,grid.y,perBlockX,perBlockY);
    printf(" uiWA %d uiWB %d \n",uiWA,uiWB);

    // volatile GPUGlobals* gpuGlobals;
    // initializer(&gpuGlobals);
    CUDA_SAFE_CALL(cudaDeviceSetLimit(cudaLimitMallocHeapSize,1<<25));

    printf("Device %d: \"%s\" with Compute %d.%d capability\n", devID, props.name, props.major, props.minor);

    // set seed for rand()
    srand(2006);


    // allocate host memory for matrices A and B
    unsigned int size_A = uiWA * uiHA;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float* h_A =NULL; //(float*)malloc(mem_size_A);
    cudaMallocHost(&h_A,mem_size_A);
    unsigned int size_B = uiWB * uiHB;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float* h_B = NULL;//(float*)malloc(mem_size_B);
    cudaMallocHost(&h_B,mem_size_B);

    // initialize host memory
    randomInit(h_A, size_A); write_mtx("mtx_a",h_A,mem_size_A);
    randomInit(h_B, size_B);
    write_mtx("mtx_b_orig",h_B,mem_size_B);
    fprintf(stderr,"1\n");
    float* h_B_t = (float*)malloc(mem_size_B);
    transpose(h_B,h_B_t,uiHB,uiWB);
    write_mtx("mtx_b",h_B_t,mem_size_B);

    // allocate device memory
    float* d_A, *d_B, *d_C;
    unsigned int size_C = uiWC * uiHC;
    unsigned int mem_size_C = sizeof(float) * size_C;

    printf("memsize_C=%d\n",mem_size_C);

    // allocate host memory for the result
    float* h_C      = NULL;//(float*) malloc(mem_size_C);
    cudaMallocHost(&h_C,mem_size_C);
    float* h_CUBLAS = (float*) malloc(mem_size_C);



    unlink("mtx_c");
    //unlink("mtx_c_orig");

    // kernel warmup
    // create and start timer

    checkCudaErrors(cudaMalloc((void**) &d_A, mem_size_A));
    checkCudaErrors(cudaMalloc((void**) &d_B, mem_size_B));
    checkCudaErrors(cudaMalloc((void**) &d_C, mem_size_C));

    char* num_iter= getenv("NUM_ITER");
    int NUM_ITERATIONS= (num_iter==NULL)?1: atoi(num_iter);

    double res_cuda_data=0;
    double res_cuda_kernel=0;
    double total_time_cuda=0;
    for(int zzz=0;zzz<NUM_ITERATIONS;zzz++){
        double time_before_cuda=_timestamp();

        int fd=open("mtx_a",O_RDONLY);
        if (fd<0) {
            perror("cant open mtx_a\n");
            exit(-1);
        }
        if(read(fd, h_A,mem_size_A)!=mem_size_A) {
            perror("cant read\n");
            exit(-1);
        }
        close(fd);
        fd=open("mtx_b_orig",O_RDONLY);
        if (fd<0) {
            perror("cant open mtx_b_orig\n");
            exit(-1);
        }
        if(read(fd, h_B,mem_size_B)!=mem_size_B) {
            perror("cant read\n");
            exit(-1);
        }
        close(fd);
        // copy host memory to device
        checkCudaErrors(cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice) );
        checkCudaErrors(cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice) );

        total_time_cuda+=(_timestamp()-time_before_cuda);

        fprintf(stderr,"CUDAMemory copy and file read: %.0f\n",(_timestamp()-time_before_cuda)/1000);
        res_cuda_data+=(_timestamp()-time_before_cuda);

        cudaEvent_t e_b; cudaEventCreate(&e_b);
        cudaEvent_t e_e; cudaEventCreate(&e_e);
        cudaEvent_t e_m; cudaEventCreate(&e_m);

        // execute the warmup kernel

        //matrixMulCUDA<32><<<gridCUDA,threads,0,0>>>(d_C,d_A,d_B,uiWA,uiWB);
        //cudaDeviceSynchronize();

        time_before_cuda=_timestamp();
        //  for(int i=0;i<5;i++){
        cudaEventRecord(e_b);
        printf("MATRIX\n");
        matrixMulCUDA<32><<<gridCUDA,threads,0,0>>>(d_C,d_A,d_B,uiWA,uiWB);
        printf("MATRIX DONE\n");

        cudaDeviceSynchronize();
        double time_kernel_only=_timestamp()-time_before_cuda;
        cudaEventRecord(e_e);
        //  }
        checkCudaErrors(cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost) );
        cudaEventRecord(e_m);
        write_mtx("mtx_c_orig",h_C,mem_size_C);
        double time_kernel_copyback= _timestamp()-time_before_cuda;
        float only_kernel=0;
        cudaEventElapsedTime(&only_kernel,e_b,e_e);
        float only_memcpy=0;
        cudaEventElapsedTime(&only_memcpy,e_e,e_m);

        total_time_cuda+=time_kernel_copyback;
        fprintf(stderr,"CUDAtime=%0.f kernel=%.0f memcpy=%.0f filecopy=%.0f gflop %0.3f\n",total_time_cuda/1000, only_kernel,only_memcpy, time_kernel_copyback/1000-only_memcpy-only_kernel, ((double)uiHA*uiWA*uiWB*2)/(1<<30)/(total_time_cuda/1e6) );

        res_cuda_data+=(time_kernel_copyback-1000*only_kernel);
        res_cuda_kernel=only_kernel*1000;
    }

    double res_cuda=total_time_cuda;

    cudaStream_t s[4];
    cudaStreamCreate(&s[0]);
    cudaStreamCreate(&s[1]);
    cudaStreamCreate(&s[2]);
    cudaStreamCreate(&s[3]);
    cudaEvent_t e_b; cudaEventCreate(&e_b);
    cudaEvent_t e_e; cudaEventCreate(&e_e);
    cudaEvent_t e_m; cudaEventCreate(&e_m);

    gridCUDA.y=gridCUDA.y/2;

    total_time_cuda=0;
    for(int zzz=0;zzz<NUM_ITERATIONS;zzz++){
        double time_before_cuda=_timestamp();

        int fd=open("mtx_a",O_RDONLY);
        if (fd<0) { perror("cant open mtx_a\n"); exit(-1);}

        int fd1=open("mtx_b_orig",O_RDONLY);
        if (fd1<0) { perror("cant open mtx_b_orig\n"); exit(-1);}

#define OVERLAPS 2
        //for(int x=0;x<OVERLAPS;++)
        if(read(fd1, h_B ,mem_size_B)!=mem_size_B) {
            perror("cant read\n");
            exit(-1);
        }
        checkCudaErrors(cudaMemcpyAsync(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice,s[0]) );

        int tileA=mem_size_A/OVERLAPS;
        int tileC=mem_size_C/OVERLAPS;
        int f=0;
        for(int y=0;y<OVERLAPS;y++){
            int offset=mem_size_A/OVERLAPS*y;

            if(pread(fd, ((char*)h_A)+offset, tileA,offset)!=tileA) {
                perror("cant read\n");
                exit(-1);
            }

            checkCudaErrors(cudaMemcpyAsync(((char*)d_A)+offset,  ((char*)h_A)+offset, tileA, cudaMemcpyHostToDevice,s[y]) );

            cudaEventRecord(e_b,s[y]);
            matrixMulCUDA<32><<<gridCUDA,threads,0,s[y]>>>(d_C+tileC*y/4,d_A+tileA*y/4,d_B,uiWA,uiWB);
            cudaEventRecord(e_e,s[y]);
            checkCudaErrors(cudaMemcpyAsync(h_C+tileC*y/4, d_C+tileC*y/4, tileC, cudaMemcpyDeviceToHost,s[y]) );
            cudaEventRecord(e_m,s[y]);

            if(y!=0){
                checkCudaErrors(cudaStreamSynchronize(s[y-1]));
                f=write_mtx("mtx_c_orig_tiled",h_C+tileC/4*(y-1),tileC,tileC*(y-1),0,f);
            }
        }
        checkCudaErrors(cudaStreamSynchronize(s[OVERLAPS-1]));
        write_mtx("mtx_c_orig_tiled",h_C+(OVERLAPS-1)*tileC/4,tileC,(OVERLAPS-1)*tileC,1,f);


        double time_kernel_copyback= _timestamp()-time_before_cuda;
        float only_kernel=0;
        cudaEventElapsedTime(&only_kernel,e_b,e_e);
        float only_memcpy=0;
        cudaEventElapsedTime(&only_memcpy,e_e,e_m);

        total_time_cuda+=time_kernel_copyback;

        fprintf(stderr,"CUDAtime=%0.f kernel=%.0f memcpy=%.0f gflop %0.3f\n",total_time_cuda/1000, only_kernel,only_memcpy,  ((double)uiHA*uiWA*uiWB*2)/(1<<30)/(total_time_cuda/1e6) );

        close(fd);
        close(fd1);
    }
    double res_tuned=total_time_cuda;


    double c_open, c_rw, c_close;
    c_open=c_rw=c_close=0;


    double total_time=0;
    // fprintf(stderr, "GPUFS open: %.0f, rw %.0f, close %.0f usec\n",c_open,c_rw,c_close);
    //     fprintf(stderr,"kernel is complete\n");
    //     fprintf(stderr,"Max pending requests: %d\n",max_req);
    //     fprintf(stderr,"Transfer time - not including sync: %.3f\n",transfer_time);





    // stop and destroy timer
    fprintf(stderr,"GPUFS >>>Total time=%0.f Gflops= %.3f\n", total_time/1000,((double)uiHA*uiWA*uiWB*2)/(1<<30)/(total_time/1e6));


    //  delete gpuGlobals;

    PRINT_MALLOC;
    PRINT_FREE;
    PRINT_PAGE_ALLOC_RETRIES;
    PRINT_LOCKLESS_SUCCESS;
    PRINT_WRONG_FILE_ID;

    PRINT_RT_MALLOC;
    PRINT_RT_FREE;
    PRINT_HT_MISS;
    PRINT_PRECLOSE_PUSH;
    PRINT_PRECLOSE_FETCH;
    PRINT_HT_HIT;
    PRINT_FLUSHED_READ;
    PRINT_FLUSHED_WRITE;
    PRINT_TRY_LOCK_FAILED;
    char fn[]="mtx_c";
    fn[0]='0';
    int fd=open(fn,O_RDONLY);
    if (fd<0) {
        perror("cant open mtx_c\n");
        exit(-1);
    }
    if(read(fd, h_C,mem_size_C)!=mem_size_C) {
        perror("cant read\n");
        exit(-1);
    }
    close(fd);
    fd=open("mtx_c_orig_tiled",O_RDONLY);
    if (fd<0) {
        perror("cant open mtx_c_orig\n");
        exit(-1);
    }
    if(read(fd, h_CUBLAS,mem_size_C)!=mem_size_C) {
        perror("cant read orig\n");
        exit(-1);
    }
    close(fd);




    printf("Comparing CUBLAS & Host results\n");
    bool resCUBLAS = sdkCompareL2fe(h_C, h_CUBLAS, size_C, 1.0e-6f);
    if (resCUBLAS != true) {
        printDiff(h_C, h_CUBLAS, uiWC, uiHC, 10000, 1.0e-5f);
    }
    fprintf(stderr,"CUBLAS compares %s\n\n", (true == resCUBLAS) ? "OK" : "FAIL");

#define FLOP(t) ((double)uiHA*uiWA*uiWB*2)/(1<<30)/(t/1e6)

    fprintf(stderr,"RESULTS: %d %d %d %d %d %d  %.0f %.0f %.0f %.3f %.3f %.3f %.0f %.0f %.3f \n",uiHA,uiWA,uiWB,uiHA*uiWA,uiWA*uiWB,uiHA*uiWB, res_cuda,res_tuned,total_time,FLOP(res_cuda),FLOP(res_tuned),FLOP(total_time), res_cuda_data, res_cuda_kernel, res_cuda_data/res_cuda_kernel);

    // clean up memory
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));

    //   cudaDeviceReset();
}

// Allocates a matrix with random float entries.

void randomInit(float* data, int size)
{
    printf("size: %d\n",size);
    for (int i = 0; i < size; ++i)
        data[i] = rand() / (float)RAND_MAX;
    //data[i]=1;

}

void transpose(float*data, float* newData, int hight, int width){
    for(int i=0;i<hight;i++){
        for( int j=0;j<width;j++){
            newData[j*hight+i]=data[i*width+j];
        }
    }
}

void printDiff(float *data1, float *data2, int width, int height, int iListLength, float fListTol)
{
    shrLog("Listing first %d Differences > %.6f...\n", iListLength, fListTol);
    int i,j,k;
    int error_count=0;
    for (j = 0; j < height; j++)
    {
        if (error_count < iListLength) {
            shrLog("\n  Row %d:\n", j);
        }
        for (i = 0; i < width; i++) {
            k = j * width + i;
            float fDiff = std::fabs(data1[k] - data2[k]);
            if (fDiff > fListTol) {
                if (error_count < iListLength) {
                    shrLog("    Loc(%d,%d)\tCPU=%.5f\tGPU=%.5f\tDiff=%.6f\n", i, j, data1[k], data2[k], fDiff);
                }
                error_count++;
            }
        }
    }
    shrLog(" \n  Total Errors = %d\n\n", error_count);
}
