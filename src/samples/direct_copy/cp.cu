#include <gloop/gloop.h>

__device__ void perform_copy(gloop::DeviceLoop<>* loop, uchar* scratch, int zfd, int zfd1, size_t me, size_t filesize)
{
    if (me < filesize) {
        size_t toRead = min((size_t)GLOOP_SHARED_PAGE_SIZE, (size_t)(filesize - me));
        gloop::fs::direct::read(loop, zfd, me, toRead, scratch, [=](gloop::DeviceLoop<>* loop, int read) {
            if (toRead != read) {
                assert(NULL);
            }

            gloop::fs::direct::write(loop, zfd1, me, toRead, scratch, [=](gloop::DeviceLoop<>* loop, int written) {
                if (toRead != written) {
                    assert(NULL);
                }
                perform_copy(loop, scratch, zfd, zfd1, me + GLOOP_SHARED_PAGE_SIZE * loop->logicalGridDim().x, filesize);
            });
        });
        return;
    }

    gloop::fs::close(loop, zfd, [=](gloop::DeviceLoop<>* loop, int err) {
        gloop::fs::close(loop, zfd1, [=](gloop::DeviceLoop<>* loop, int err) {
        });
    });
}

__device__ void gpuMain(gloop::DeviceLoop<>* loop, char* src, char* dst, uint8_t* data)
{
    uint8_t* scratch = data + loop->logicalBlockIdx().x * GLOOP_SHARED_PAGE_SIZE;
    gloop::fs::open(loop, src, O_RDONLY, [=](gloop::DeviceLoop<>* loop, int zfd) {
        gloop::fs::open(loop, dst, O_WRONLY | O_CREAT, [=](gloop::DeviceLoop<>* loop, int zfd1) {
            gloop::fs::fstat(loop, zfd, [=](gloop::DeviceLoop<>* loop, int filesize) {
                size_t me = loop->logicalBlockIdx().x * GLOOP_SHARED_PAGE_SIZE;
                perform_copy(loop, scratch, zfd, zfd1, me, filesize);
            });
        });
    });
}
