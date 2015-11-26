
#include "fs_calls.cu.h"
#include "gloop.h"
__device__ int OK;
__shared__ int zfd,zfd1, zfd2, close_ret;

template<typename Callback>
__device__ void perform_copy(uchar* scratch, size_t me, size_t filesize, const Callback& callback)
{
    if (me < filesize) {
        int toRead=min((unsigned int)FS_BLOCKSIZE,(unsigned int)(filesize-me));
        gloop::read(zfd, me, toRead, scratch, [=](size_t read) {
            if (toRead!=read) {
                assert(NULL);
            }

            gloop::write(zfd1,me,toRead,scratch, [=](size_t written) {
                if (toRead!=written) {
                    assert(NULL);
                }
                perform_copy(scratch, me + FS_BLOCKSIZE*gridDim.x, filesize, callback);
            });
        });
        return;
    }
    callback();
}

__device__ LAST_SEMAPHORE sync_sem;
__global__ void test_cpy(char* src, char* dst)
{
    __shared__ uchar* scratch;
    BEGIN_SINGLE_THREAD
        scratch=(uchar*)malloc(FS_BLOCKSIZE);
        GPU_ASSERT(scratch!=NULL);
    END_SINGLE_THREAD

    gloop::open(src,O_GRDONLY, [=](int zfd) {
        gloop::open(dst,O_GWRONCE, [=](int zfd1) {
            gloop::fstat(zfd, [=](size_t filesize) {
                size_t me=blockIdx.x*FS_BLOCKSIZE;
                perform_copy(scratch, me, filesize, [=] () {
                    gloop::close(zfd, [=](int err) {
                        gloop::close(zfd1, [=](int err) {
                        });
                    });
                });
            });
        });
    });
}

void init_device_app(){
    CUDA_SAFE_CALL(cudaDeviceSetLimit(cudaLimitMallocHeapSize,1<<30));
}

void init_app()
{
    void* d_OK;
    CUDA_SAFE_CALL(cudaGetSymbolAddress(&d_OK,OK));
    CUDA_SAFE_CALL(cudaMemset(d_OK,0,sizeof(int)));
    // INITI LOCK
    void* inited;


    CUDA_SAFE_CALL(cudaGetSymbolAddress(&inited,sync_sem));
    CUDA_SAFE_CALL(cudaMemset(inited,0,sizeof(LAST_SEMAPHORE)));
}

double post_app(double time, int trials){
    int res;
    CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&res,OK,sizeof(int),0,cudaMemcpyDeviceToHost));
    if(res!=0) fprintf(stderr,"Test Failed, error code: %d \n",res);
    else  fprintf(stderr,"Test Success\n");

    return 0;
}

