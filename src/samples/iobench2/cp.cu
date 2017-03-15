#include <gloop/gloop.h>

__device__ void performCopy(gloop::DeviceLoop<>* loop, uchar* scratch, int zfd, size_t me, size_t filesize, int trials, int ioSize, int loopCount)
{
    volatile int res = 0;
    for (int i = 0; i < loopCount; ++i) {
        res += i;
    }

    if (me < filesize) {
        size_t toRead = min((size_t)GLOOP_SHARED_PAGE_SIZE, (size_t)(filesize - me));
        gloop::fs::read(loop, zfd, me, toRead, scratch, [=](gloop::DeviceLoop<>* loop, int read) {
            if (toRead != read) {
                assert(NULL);
            }

            performCopy(loop, scratch, zfd, me + GLOOP_SHARED_PAGE_SIZE * loop->logicalGridDim().x, filesize, trials, ioSize, loopCount);
        });
        return;
    }

    gloop::fs::close(loop, zfd, [=](gloop::DeviceLoop<>* loop, int err) {});
}

__device__ void gpuMain(gloop::DeviceLoop<>* loop, char* src, int trials, int ioSize, int loopCount)
{
    __shared__ uchar* scratch;

    BEGIN_SINGLE_THREAD
        scratch = (uchar*)malloc(GLOOP_SHARED_PAGE_SIZE);
        GLOOP_ASSERT(scratch != NULL);
    END_SINGLE_THREAD

    gloop::fs::open(loop, src, O_RDONLY, [=](gloop::DeviceLoop<>* loop, int zfd) {
        gloop::fs::fstat(loop, zfd, [=](gloop::DeviceLoop<>* loop, int filesize) {
            size_t me = loop->logicalBlockIdx().x * GLOOP_SHARED_PAGE_SIZE;
            performCopy(loop, scratch, zfd, me, filesize, trials, ioSize, loopCount);
        });
    });
}
