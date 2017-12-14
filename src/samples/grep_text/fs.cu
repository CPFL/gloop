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

// INCLUDING CODE INLINE - change later
#include <gloop/gloop.h>
#include <gloop/benchmark.h>
//



#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>

void __device__ grep_text(gloop::DeviceLoop<>* loop, char* src, char* out, char* dbs);
void init_device_app();
void init_app();

char*  update_filename(const char* h_filename){
	int n=strlen(h_filename);
	assert(n>0);
	if (n>GLOOP_FILENAME_SIZE) {
		fprintf(stderr,"Filname %s too long, should be only %d symbols including \\0",h_filename,GLOOP_FILENAME_SIZE);
		exit (-1);
	}
	char* d_filename;
	CUDA_SAFE_CALL(cudaMalloc(&d_filename,n+1));
	CUDA_SAFE_CALL(cudaMemcpy(d_filename, h_filename, n+1,cudaMemcpyHostToDevice));
	return d_filename;
}

#include <assert.h>

#define TRIALS 1.0
int main( int argc, char** argv)
{


	if(argc<6) {
		fprintf(stderr,"<id> <blocks> <physblocks> <threads> f1 f2 ... f_#files\n");
		return -1;
	}
	int id=atoi(argv[1]);
	int nblocks=atoi(argv[2]);
	int physnblocks=atoi(argv[3]);
	int nthreads=atoi(argv[4]);

	fprintf(stderr," id: %d blocks %d threads %d\n",id, nblocks, nthreads);

	int num_files=argc-1-4;
	char** d_filenames=NULL;


	double total_time=0;
//	int scratch_size=128*1024*1024*4;

    {
        dim3 blocks(nblocks);
        dim3 physblocks(physnblocks);

        std::unique_ptr<gloop::HostLoop> hostLoop = gloop::HostLoop::create(GLOOP_DEVICE);
        std::unique_ptr<gloop::HostContext> hostContext = gloop::HostContext::create(*hostLoop, physblocks);

        {
            std::lock_guard<gloop::HostLoop::KernelLock> lock(hostLoop->kernelLock());
            init_device_app();
            init_app();

            if (num_files>0){
                d_filenames=(char**)malloc(sizeof(char*)*num_files);
                for(int i=0;i<num_files;i++){
                    d_filenames[i]=update_filename(argv[i+5]);
                    fprintf(stderr,"file -%s\n",argv[i+5]);
                }
            }
        }
        gloop::Benchmark benchmark;
        benchmark.begin();
        hostLoop->launch(*hostContext, blocks, nthreads, [] __device__ (gloop::DeviceLoop<>* loop, char* src, char* out, char* dbs) {
            grep_text(loop, src, out, dbs);
        }, d_filenames[0], d_filenames[1], d_filenames[2]);
        benchmark.end();
        fprintf(stderr, "[%d] ", id);
        benchmark.report(stderr);
    }
	if (d_filenames) free(d_filenames);
	return 0;
}
