/*
 * This expermental software is provided AS IS.
 * Feel free to use/modify/distribute,
 * If used, please retain this disclaimer and cite
 * "GPUfs: Integrating a file system with GPUs",
 * M Silberstein,B Ford,I Keidar,E Witchel
 * ASPLOS13, March 2013, Houston,USA
 */


#include <cuda.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <math.h>
#include <assert.h>

#include <gloop/gloop.h>
#include <gloop/benchmark.h>

#include "img.cuh"

void init_device_app();
void init_app();
double post_app(double total_time, float trials);

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
    if (n>GLOOP_FILENAME_SIZE) {
        fprintf(stderr,"Filname %s too long, should be only %d symbols including \\0",h_filename,GLOOP_FILENAME_SIZE);
        exit (-1);
    }
    char* d_filename;
    GLOOP_CUDA_SAFE_CALL(cudaMalloc(&d_filename,n+1));
    GLOOP_CUDA_SAFE_CALL(cudaMemcpy(d_filename, h_filename, n+1,cudaMemcpyHostToDevice));
    return d_filename;
}

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

	if(argc<5) {
		fprintf(stderr,"<id> <blocks> <threads> f1 f2 ... f_#files\n");
		return -1;
	}
	int id=atoi(argv[1]);
	int nblocks=atoi(argv[2]);
	int nthreads=atoi(argv[3]);

	fprintf(stderr," id: %d blocks %d threads %d\n",id, nblocks, nthreads);

	int num_files=argc-1-3;
	char** d_filenames=NULL;


	double total_time=0;
    {
        dim3 blocks(nblocks);

        std::unique_ptr<gloop::HostLoop> hostLoop = gloop::HostLoop::create(0);
        std::unique_ptr<gloop::HostContext> hostContext = gloop::HostContext::create(*hostLoop, blocks);

        {
            std::lock_guard<gloop::HostLoop::KernelLock> lock(hostLoop->kernelLock());
            init_device_app();
            init_app();


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
        hostLoop->launch(*hostContext, blocks, nthreads, [] __device__ (gloop::DeviceLoop<>* loop, char* src, int src_row_len, int num_db_files, float match_threshold, int start_offset, char* out, char* out2, char* out3, char* out4, char* out5, char* out6, char* out7) {
            img_gpu(loop, src, src_row_len, num_db_files, match_threshold, start_offset, out, out2, out3, out4, out5, out6, out7);
        },
        d_filenames[0],
        GREP_ROW_WIDTH, num_files - 2, match_threshold, 0,
        d_filenames[1],
        d_filenames[2], // db0
        d_filenames[3],d_filenames[4],
        d_filenames[5],d_filenames[6],d_filenames[7]);


        benchmark.end();
        printf("[%d] ", id);
        benchmark.report(stderr);
    }
	if (d_filenames) free(d_filenames);
	return 0;
}
