#include <gloop/gloop.h>

__device__ void perform_copy(gloop::DeviceLoop* loop, uchar* scratch, int zfd, int zfd1, size_t me, size_t filesize)
{
    if (me < filesize) {
        size_t toRead = min((size_t)GLOOP_SHARED_PAGE_SIZE, (size_t)(filesize - me));
        gloop::fs::read(loop, zfd, me, toRead, scratch, [=](gloop::DeviceLoop* loop, int read) {
            if (toRead != read) {
                assert(NULL);
            }

            gloop::fs::write(loop, zfd1, me, toRead, scratch, [=](gloop::DeviceLoop* loop, int written) {
                if (toRead != written) {
                    assert(NULL);
                }
                perform_copy(loop, scratch, zfd, zfd1, me + GLOOP_SHARED_PAGE_SIZE * gloop::logicalGridDim.x, filesize);
            });
        });
        return;
    }

    gloop::fs::close(loop, zfd, [=](gloop::DeviceLoop* loop, int err) {
        gloop::fs::close(loop, zfd1, [=](gloop::DeviceLoop* loop, int err) {
        });
    });
}

__device__ void gpuMain(gloop::DeviceLoop* loop, char* src, char* dst)
{
    __shared__ uchar* scratch;

    BEGIN_SINGLE_THREAD
        scratch = (uchar*)malloc(GLOOP_SHARED_PAGE_SIZE);
        GLOOP_ASSERT(scratch != NULL);
    END_SINGLE_THREAD

    gloop::fs::open(loop, src, O_RDONLY, [=](gloop::DeviceLoop* loop, int zfd) {
        gloop::fs::open(loop, dst, O_WRONLY | O_CREAT, [=](gloop::DeviceLoop* loop, int zfd1) {
            gloop::fs::fstat(loop, zfd, [=](gloop::DeviceLoop* loop, int filesize) {
                size_t me = gloop::logicalBlockIdx.x * GLOOP_SHARED_PAGE_SIZE;
                perform_copy(loop, scratch, zfd, zfd1, me, filesize);
            });
        });
    });
}
