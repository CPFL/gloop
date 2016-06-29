#include <gloop/gloop.h>

struct PairedBuffers {
    uchar* buffers[2];
    int* guard;
};

__forceinline__ __device__ void processScratch(uchar* buffer, size_t size)
{
    // Processing the scratch.
}

__forceinline__ __device__ void performCopyMiddle(gloop::DeviceLoop<>* loop, PairedBuffers pairedBuffers, int zfd, int zfd1, size_t prevMe, size_t me, size_t filesize, size_t toWrite, int readBuffer)
{
    uchar* scratch = pairedBuffers.buffers[readBuffer];
    if (me < filesize) {
        size_t toRead = min((size_t)GLOOP_SHARED_PAGE_SIZE, (size_t)(filesize - me));

        auto next = [=](gloop::DeviceLoop<>* loop) {
            __shared__ bool goTo;
            BEGIN_SINGLE_THREAD
            {
                __threadfence_block();
                goTo = false;
                if ((*pairedBuffers.guard)++ == 1) {
                    *pairedBuffers.guard = 0;
                    goTo = true;
                }
            }
            END_SINGLE_THREAD
            if (goTo) {
                performCopyMiddle(loop, pairedBuffers, zfd, zfd1, me, me + GLOOP_SHARED_PAGE_SIZE * gloop::logicalGridDim.x, filesize, toRead, readBuffer ^ 1);
            }
        };

        // Perform read.

        gloop::fs::read(loop, zfd, me, toRead, pairedBuffers.buffers[readBuffer ^ 1], [=](gloop::DeviceLoop<>* loop, int read) {
            if (toRead != read) {
                assert(NULL);
            }
            next(loop);
        });

        processScratch(scratch, toWrite);

        // And then, write the scratch.
        gloop::fs::write(loop, zfd1, prevMe, toWrite, scratch, [=](gloop::DeviceLoop<>* loop, int written) {
            if (toWrite != written) {
                assert(NULL);
            }
            next(loop);
        });
        return;
    }

    processScratch(scratch, toWrite);

    // And then, write the scratch and close the file.
    gloop::fs::write(loop, zfd1, prevMe, toWrite, scratch, [=](gloop::DeviceLoop<>* loop, int written) {
        if (toWrite != written) {
            assert(NULL);
        }
        gloop::fs::close(loop, zfd, [=](gloop::DeviceLoop<>* loop, int err) {
            gloop::fs::close(loop, zfd1, [=](gloop::DeviceLoop<>* loop, int err) {
            });
        });
    });
}

__device__ void performCopyFirst(gloop::DeviceLoop<>* loop, PairedBuffers pairedBuffers, int zfd, int zfd1, size_t me, size_t filesize)
{
    if (me < filesize) {
        size_t toRead = min((size_t)GLOOP_SHARED_PAGE_SIZE, (size_t)(filesize - me));

        gloop::fs::read(loop, zfd, me, toRead, pairedBuffers.buffers[0], [=](gloop::DeviceLoop<>* loop, int read) {
            if (toRead != read) {
                assert(NULL);
            }
            performCopyMiddle(loop, pairedBuffers, zfd, zfd1, me, me + GLOOP_SHARED_PAGE_SIZE * gloop::logicalGridDim.x, filesize, toRead, 0);
        });
        return;
    }

    gloop::fs::close(loop, zfd, [=](gloop::DeviceLoop<>* loop, int err) {
        gloop::fs::close(loop, zfd1, [=](gloop::DeviceLoop<>* loop, int err) {
        });
    });
}

__device__ void gpuMain(gloop::DeviceLoop<>* loop, char* src, char* dst)
{
    __shared__ PairedBuffers pairedBuffers;

    BEGIN_SINGLE_THREAD
    {
        uchar* allBuffers = (uchar*)malloc(GLOOP_SHARED_PAGE_SIZE * 2 + sizeof(int));
        pairedBuffers.buffers[0] = allBuffers;
        pairedBuffers.buffers[1] = allBuffers + GLOOP_SHARED_PAGE_SIZE;
        pairedBuffers.guard = (int*)(allBuffers + GLOOP_SHARED_PAGE_SIZE * 2);
        *pairedBuffers.guard = 0;
        GLOOP_ASSERT(allBuffers != NULL);
    }
    END_SINGLE_THREAD

    gloop::fs::open(loop, src, O_RDONLY, [=](gloop::DeviceLoop<>* loop, int zfd) {
        gloop::fs::open(loop, dst, O_WRONLY | O_CREAT, [=](gloop::DeviceLoop<>* loop, int zfd1) {
            gloop::fs::fstat(loop, zfd, [=](gloop::DeviceLoop<>* loop, int filesize) {
                size_t me = loop->logicalBlockIdx().x * GLOOP_SHARED_PAGE_SIZE;
                performCopyFirst(loop, pairedBuffers, zfd, zfd1, me, filesize);
            });
        });
    });
}
