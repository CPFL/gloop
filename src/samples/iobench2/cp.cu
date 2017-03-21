#include <gloop/gloop.h>

__device__ void performCopy(gloop::DeviceLoop<>* loop, uchar* scratch, int zfd, size_t filesize, int trials, int ioSize, int loopCount, int count)
{
    volatile int res = 0;
    for (int i = 0; i < loopCount; ++i) {
        res += i;
    }

    if (count < trials) {
        gloop::fs::read(loop, zfd, 0, ioSize, scratch, [=](gloop::DeviceLoop<>* loop, int read) {
            if (ioSize != read) {
                assert(NULL);
            }

            performCopy(loop, scratch, zfd, filesize, trials, ioSize, loopCount, count + 1);
        });
        return;
    }

    gloop::fs::close(loop, zfd, [=](gloop::DeviceLoop<>* loop, int err) {});
}

__device__ void gpuMain(gloop::DeviceLoop<>* loop, char* src, int trials, int ioSize, int loopCount)
{
    __shared__ uchar* scratch;

    BEGIN_SINGLE_THREAD
    {
        scratch = (uchar*)malloc(ioSize);
        GLOOP_ASSERT(scratch != NULL);
    }
    END_SINGLE_THREAD

    gloop::fs::open(loop, src, O_RDONLY, [=](gloop::DeviceLoop<>* loop, int zfd) {
        gloop::fs::fstat(loop, zfd, [=](gloop::DeviceLoop<>* loop, int filesize) {
            performCopy(loop, scratch, zfd, filesize, trials, ioSize, loopCount, 0);
        });
    });
}
