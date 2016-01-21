#include "fs_debug.cu.h"
#include "fs_initializer.cu.h"

#include <gloop/gloop.h>
__device__ int OK;

__device__ void perform_copy(gloop::DeviceLoop* loop, uchar* scratch, int zfd, int zfd1, size_t me, size_t filesize)
{
    if (me < filesize) {
        size_t toRead = min((size_t)FS_BLOCKSIZE, (size_t)(filesize-me));
        gloop::fs::read(loop, zfd, me, toRead, scratch, [=](gloop::DeviceLoop* loop, int read) {
            if (toRead != read) {
                assert(NULL);
            }

#if 0
            gloop::fs::write(loop, zfd1, me, toRead, scratch, [=](gloop::DeviceLoop* loop, int written) {
                if (toRead != written) {
                    assert(NULL);
                }
                perform_copy(loop, scratch, zfd, zfd1, me + FS_BLOCKSIZE * gridDim.x, filesize);
            });
#endif
        });
        return;
    }

    gloop::fs::close(loop, zfd, [=](gloop::DeviceLoop* loop, int err) {
        gloop::fs::close(loop, zfd1, [=](gloop::DeviceLoop* loop, int err) {
        });
    });
}

__device__ LAST_SEMAPHORE sync_sem;
__device__ void test_cpy(gloop::DeviceLoop* loop, char* src, char* dst)
{
    __shared__ uchar* scratch;

    BEGIN_SINGLE_THREAD
        scratch=(uchar*)malloc(FS_BLOCKSIZE);
        GPU_ASSERT(scratch!=NULL);
    END_SINGLE_THREAD

    gloop::fs::open(loop, src, O_GRDONLY, [=](gloop::DeviceLoop* loop, int zfd) {
        gloop::fs::open(loop, dst, O_GWRONCE, [=](gloop::DeviceLoop* loop, int zfd1) {
            gloop::fs::fstat(loop, zfd, [=](gloop::DeviceLoop* loop, int filesize) {
                size_t me = blockIdx.x * FS_BLOCKSIZE;
                perform_copy(loop, scratch, zfd, zfd1, me, filesize);
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
