#include "fs_debug.cu.h"
#include "fs_initializer.cu.h"

#include "gloop.h"
__device__ int OK;

__device__ void perform_copy(gloop::DeviceLoop* loop, uchar* scratch, int zfd, int zfd1, size_t me, size_t filesize);

__device__ void close2(gloop::DeviceLoop* loop, Close2Data* data)
{
}

__device__ void close1(gloop::DeviceLoop* loop, Close1Data* data)
{
    int zfd1 = data->zfd1;
    gloop::close(loop, zfd1, (struct Close2Data) {
        Close2
    });
}

__device__ void write1(gloop::DeviceLoop* loop, Write1Data* data)
{
    uchar* scratch = data->scratch;
    int zfd = data->zfd;
    int zfd1 = data->zfd1;
    size_t filesize = data->filesize;
    size_t me = data->me;
    size_t toRead = data->toRead;
    size_t written = data->written;
    if (toRead != written) {
        assert(NULL);
    }
    perform_copy(loop, scratch, zfd, zfd1, me + FS_BLOCKSIZE * gridDim.x, filesize);
}

__device__ void read1(gloop::DeviceLoop* loop, Read1Data* data)
{
    uchar* scratch = data->scratch;
    int zfd = data->zfd;
    int zfd1 = data->zfd1;
    size_t filesize = data->filesize;
    size_t me = data->me;
    size_t toRead = data->toRead;
    size_t read = data->read;
    if (toRead != read) {
        assert(NULL);
    }

    gloop::write(loop, zfd1, me, toRead, scratch, (struct Write1Data) {
        Write1,
        scratch,
        zfd,
        zfd1,
        filesize,
        me,
        toRead,
        {}
    });
}

__device__ void perform_copy(gloop::DeviceLoop* loop, uchar* scratch, int zfd, int zfd1, size_t me, size_t filesize)
{
    if (me < filesize) {
        size_t toRead = min((size_t)FS_BLOCKSIZE, (size_t)(filesize-me));
        gloop::read(loop, zfd, me, toRead, scratch, (struct Read1Data) {
            Read1,
            scratch,
            zfd,
            zfd1,
            filesize,
            me,
            toRead,
            {}
        });
        return;
    }
    gloop::close(loop, zfd, (struct Close1Data) {
        Close1,
        zfd1
    });
}

__device__ void fstat1(gloop::DeviceLoop* loop, Fstat1Data* data)
{
    uchar* scratch = data->scratch;
    int zfd = data->zfd;
    int zfd1 = data->zfd1;
    size_t filesize = data->filesize;
    size_t me = blockIdx.x * FS_BLOCKSIZE;
    perform_copy(loop, scratch, zfd, zfd1, me, filesize);
}

__device__ void open2(gloop::DeviceLoop* loop, Open2Data* data)
{
    uchar* scratch = data->scratch;
    int zfd = data->zfd;
    int zfd1 = data->zfd1;
    gloop::fstat(loop, zfd, (struct Fstat1Data) {
        Fstat1,
        scratch,
        zfd,
        zfd1,
        {}
    });
}

__device__ void open1(gloop::DeviceLoop* loop, Open1Data* data)
{
    uchar* scratch = data->scratch;
    char* dst = data->dst;
    int zfd = data->zfd;
    gloop::open(loop, dst, O_GWRONCE, (struct Open2Data) {
        Open2,
        scratch,
        zfd,
        {}
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

    gloop::open(loop, src, O_GRDONLY, (struct Open1Data) {
        Open1,
        scratch,
        dst,
        {}
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

