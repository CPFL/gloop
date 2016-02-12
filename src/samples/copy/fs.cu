
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
#include <gloop/gloop.h>
#include <gloop/benchmark.h>

__device__ void gpuMain(gloop::DeviceLoop* loop, char* src, char* dst);

#define MAIN_FS_FILE

void stdavg(double *avg_time, double *avg_thpt, double* std_time, double *std_thpt, const double* times, const double total_data, int arr_len)
{
    *avg_time=*avg_thpt=*std_time=*std_thpt=0;
    int counter=0;

    for( int i=0;i<arr_len;i++){
        if (times[i]<=0) continue;

        *avg_time+=times[i];
        *avg_thpt+=((double)total_data)/times[i];
        counter++;
    }
    if (counter==0) return;
    *avg_time/=(double)counter;
    *avg_thpt/=(double)counter;

    for( int i=0;i<arr_len;i++){
        if (times[i]<=0) continue;
        *std_time=(times[i]-*avg_time)*(times[i]-*avg_time);

        double tmp=(((double)total_data)/times[i])-*avg_thpt;
        *std_thpt=tmp*tmp;
    }
    *std_time/=(double)counter;
    *std_thpt/=(double)counter;

    *std_time=sqrt(*std_time);
    *std_thpt=sqrt(*std_thpt);

}


char*  update_filename(const char* h_filename){
    int n=strlen(h_filename);
    assert(n>0);
    if (n>FILENAME_SIZE) {
        fprintf(stderr,"Filname %s too long, should be only %d symbols including \\0",h_filename,FILENAME_SIZE);
        exit (-1);
    }
    char* d_filename;
    CUDA_SAFE_CALL(cudaMalloc(&d_filename,n+1));
    CUDA_SAFE_CALL(cudaMemcpy(d_filename, h_filename, n+1,cudaMemcpyHostToDevice));
    return d_filename;
}

#include <assert.h>

// size of the output used for data staging
int output_size=FS_BLOCKSIZE;

#define MAX_TRIALS (10)
double time_res[MAX_TRIALS];

double match_threshold;
int global_devicenum;

int main( int argc, char** argv)
{

    char* threshold=getenv("GREPTH");
    match_threshold=0.01;
    if(threshold!=0) match_threshold=strtof(threshold,NULL);
    fprintf(stderr,"Match threshold is %f\n",match_threshold);

    char* gpudev=getenv("GPUDEVICE");
    global_devicenum=0;
    if (gpudev!=NULL) global_devicenum=atoi(gpudev);

    fprintf(stderr,"GPU device chosen %d\n",global_devicenum);

    if(argc<5) {
        fprintf(stderr,"<kernel_iterations> <blocks> <threads> f1 f2 ... f_#files\n");
        return -1;
    }
    int trials=atoi(argv[1]);
    assert(trials<=MAX_TRIALS);
    int nblocks=atoi(argv[2]);
    int nthreads=atoi(argv[3]);

    fprintf(stderr," iterations: %d blocks %d threads %d\n",trials, nblocks, nthreads);

    int num_files=argc-1-3;
    char** d_filenames=NULL;


    double total_time=0;
    size_t total_size;

    std::memset(time_res,0,MAX_TRIALS*sizeof(double));
    for(int i=1;i<trials+1;i++){
        dim3 blocks(nblocks);
        std::unique_ptr<gloop::HostLoop> hostLoop = gloop::HostLoop::create(global_devicenum);
        std::unique_ptr<gloop::HostContext> hostContext = gloop::HostContext::create(*hostLoop, blocks);

        {
            std::lock_guard<gloop::HostLoop::KernelLock> lock(hostLoop->kernelLock());
            CUDA_SAFE_CALL(cudaDeviceSetLimit(cudaLimitMallocHeapSize, (2 << 20) * 256));

            if (num_files>0){
                d_filenames=(char**)malloc(sizeof(char*)*num_files);
                for(int i=0;i<num_files;i++){
                    d_filenames[i]=update_filename(argv[i+4]);
                    fprintf(stderr,"file -%s\n",argv[i+4]);
                }
            }
        }

        gloop::Benchmark benchmark;
        benchmark.begin();
        {
            hostLoop->launch(*hostContext, nthreads, [=] GLOOP_DEVICE_LAMBDA (gloop::DeviceLoop* loop, thrust::tuple<char*, char*> tuple) {
                char* src;
                char* dst;
                thrust::tie(src, dst) = tuple;
                gpuMain(loop, src, dst);
            }, d_filenames[0], d_filenames[1]);
        }
        benchmark.end();
        benchmark.report();
    }
    return 0;
}



