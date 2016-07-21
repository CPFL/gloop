////////////////////////////////////////////////////////////////////////////////
// Includes
////////////////////////////////////////////////////////////////////////////////
#include "mergesort.cuh"
#include "mergesort_inlines.cuh"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
////////////////////////////////////////////////////////////////////////////////
// Defines
////////////////////////////////////////////////////////////////////////////////
#define BLOCKSIZE 256
#define ROW_LENGTH BLOCKSIZE * 4
#define ROWS 4096

struct MergeSortContext {
    float4 a;
    float4 b;
    int aidx;
    int bidx;
    int disabled;
};

// FIXME: Shared!
typedef gloop::Global LoopType;

GLOOP_VISIBILITY_HIDDEN
static __device__ void destroyContext(gloop::DeviceLoop<LoopType>* loop, MergeSortContext* context)
{
    BEGIN_SINGLE_THREAD
    {
        if (loop->isLastLogicalBlock()) {
            delete [] context;
        }
    }
    END_SINGLE_THREAD
}

GLOOP_VISIBILITY_HIDDEN
static __device__ void mergeSortPassKernel(gloop::DeviceLoop<LoopType>* loop, MergeSortContext* context, float4* result, int nrElems, int threadsPerDiv, int outidx, int Astart, int Bstart, int tid, int division, int aidx, int bidx, float4 a, float4 b, int disabled)
{
    float4* resStart = result + Astart;
    while (true) //aidx < nrElems/2)// || (bidx < nrElems/2  && (Bstart + bidx < constEndAddr[division])))
    {
        float4 nextA;
        float4 nextB;
        bool elemsLeftInA = false;
        bool elemsLeftInB = false;
        if (!disabled) {
            /**
             * For some reason, it's faster to do the texture fetches here than
             * after the merge
             */
            nextA = tex1Dfetch(tex, Astart + aidx + 1);
            nextB = tex1Dfetch(tex, Bstart + bidx + 1);

            float4 na = getLowest(a, b);
            float4 nb = getHighest(a, b);
            a = sortElem(na);
            b = sortElem(nb);
            // Now, a contains the lowest four elements, sorted
            resStart[outidx] = a;

            elemsLeftInA = (aidx + 1 < nrElems / 2); // Astart + aidx + 1 is allways less than division border
            elemsLeftInB = (bidx + 1 < nrElems / 2) && (Bstart + bidx + 1 < constStartAddr[division + 1]);

            if (elemsLeftInA) {
                if (elemsLeftInB) {
                    if (nextA.x < nextB.x) {
                        aidx += 1;
                        a = nextA;
                    } else {
                        bidx += 1;
                        a = nextB;
                    }
                } else {
                    aidx += 1;
                    a = nextA;
                }
            } else {
                if (elemsLeftInB) {
                    bidx += 1;
                    a = nextB;
                }
            }
        }
        outidx++;

        if (__syncthreads_and(!elemsLeftInA && !elemsLeftInB))
            break;

        if (gloop::loop::postTaskIfNecessary(loop, [context, result, nrElems, threadsPerDiv, outidx] (gloop::DeviceLoop<LoopType>* loop) {
            int tid = (loop->logicalBlockIdx().x * blockDim.x) + threadIdx.x;
            int division = tid / threadsPerDiv;
            int int_tid = tid - division * threadsPerDiv;
            int disabled = context[threadIdx.x].disabled;
            int Astart = 0;
            int Bstart = 0;
            if (!disabled) {
                Astart = constStartAddr[division] + int_tid * nrElems;
                Bstart = Astart + nrElems / 2;
            }
            mergeSortPassKernel(loop, context, result, nrElems, threadsPerDiv, outidx, Astart, Bstart, tid, division, context[threadIdx.x].aidx, context[threadIdx.x].bidx, context[threadIdx.x].a, context[threadIdx.x].b, disabled);
        })) {
            context[threadIdx.x].aidx = aidx;
            context[threadIdx.x].bidx = bidx;
            context[threadIdx.x].a = a;
            context[threadIdx.x].b = b;
            context[threadIdx.x].disabled = disabled;
            return;
        }
    }

    if (!disabled) {
        resStart[outidx++] = b;
    }
    destroyContext(loop, context);
}

GLOOP_VISIBILITY_HIDDEN
static __device__ void mergeSortPassInitialKernel(gloop::DeviceLoop<LoopType>* loop, float4* result, int nrElems, int threadsPerDiv)
{
    BEGIN_SINGLE_THREAD
    {
        if (!loop->scratch()) {
            loop->scratch() = (void*)new MergeSortContext[blockDim.x];
        }
    }
    END_SINGLE_THREAD
    MergeSortContext* context = (MergeSortContext*)(loop->scratch());

    int tid = (loop->logicalBlockIdx().x * blockDim.x) + threadIdx.x;
    // The division to work on
    int division = tid / threadsPerDiv;
    int disabled = 0;
    int Astart = 0;
    int Bstart = 0;
    float4 a;
    float4 b;

    if (division >= DIVISIONS) {
        disabled = 1;
    } else {
        // The block within the division
        int int_tid = tid - division * threadsPerDiv;
        Astart = constStartAddr[division] + int_tid * nrElems;

        Bstart = Astart + nrElems / 2;
        float4* resStart = &(result[Astart]);

        if (Astart >= constStartAddr[division + 1]) {
            disabled = 1;
        } else {
            if (Bstart >= constStartAddr[division + 1]) {
                for (int i = 0; i < (constStartAddr[division + 1] - Astart); i++) {
                    resStart[i] = tex1Dfetch(tex, Astart + i);
                }
                disabled = 1;
            } else {
                a = tex1Dfetch(tex, Astart);
                b = tex1Dfetch(tex, Bstart);
            }
        }
    }

    mergeSortPassKernel(loop, context, result, nrElems, threadsPerDiv, 0, Astart, Bstart, tid, division, 0, 0, a, b, disabled);
}

////////////////////////////////////////////////////////////////////////////////
// The mergesort algorithm
////////////////////////////////////////////////////////////////////////////////
float4* runMergeSort(gloop::HostLoop& hostLoop, gloop::HostContext& hostContext, int listsize, int divisions,
    float4* d_origList, float4* d_resultList,
    int* sizes, int* nullElements,
    unsigned int* origOffsets)
{
    int* startaddr = (int*)malloc((divisions + 1) * sizeof(int));
    int largestSize = -1;
    startaddr[0] = 0;
    for (int i = 1; i <= divisions; i++) {
        startaddr[i] = startaddr[i - 1] + sizes[i - 1];
        if (sizes[i - 1] > largestSize)
            largestSize = sizes[i - 1];
    }
    largestSize *= 4;

    // Setup texture
    cudaChannelFormatDesc channelDesc;
    {
        std::lock_guard<gloop::HostLoop::KernelLock> lock(hostLoop.kernelLock());
        channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
        tex.addressMode[0] = cudaAddressModeWrap;
        tex.addressMode[1] = cudaAddressModeWrap;
        tex.filterMode = cudaFilterModePoint;
        tex.normalized = false;
    }

////////////////////////////////////////////////////////////////////////////
// First sort all float4 elements internally
////////////////////////////////////////////////////////////////////////////
#ifdef MERGE_WG_SIZE_0
    const int THREADS = MERGE_WG_SIZE_0;
#else
    const int THREADS = 256;
#endif
    dim3 threads(THREADS, 1);
    int blocks = ((listsize / 4) % THREADS == 0) ? (listsize / 4) / THREADS : (listsize / 4) / THREADS + 1;
    dim3 grid(blocks, 1);
    {
        std::lock_guard<gloop::HostLoop::KernelLock> lock(hostLoop.kernelLock());
        cudaBindTexture(0, tex, d_origList, channelDesc, listsize * sizeof(float));
    }
    mergeSortFirst(hostLoop, hostContext, grid, threads, d_resultList, listsize);

    ////////////////////////////////////////////////////////////////////////////
    // Then, go level by level
    ////////////////////////////////////////////////////////////////////////////
    {
        std::lock_guard<gloop::HostLoop::KernelLock> lock(hostLoop.kernelLock());
        cudaMemcpyToSymbol(constStartAddr, startaddr, (divisions + 1) * sizeof(int));
        cudaMemcpyToSymbol(finalStartAddr, origOffsets, (divisions + 1) * sizeof(int));
        cudaMemcpyToSymbol(nullElems, nullElements, (divisions) * sizeof(int));
    }
    int nrElems = 2;
    while (true) {
        int floatsperthread = (nrElems * 4);
        int threadsPerDiv = (int)ceil(largestSize / (float)floatsperthread);
        int threadsNeeded = threadsPerDiv * divisions;
#ifdef MERGE_WG_SIZE_1
        threads.x = MERGE_WG_SIZE_1;
#else
        threads.x = 208;
#endif
        grid.x = ((threadsNeeded % threads.x) == 0) ? threadsNeeded / threads.x : (threadsNeeded / threads.x) + 1;
        if (grid.x < 8) {
            grid.x = 8;
            threads.x = ((threadsNeeded % grid.x) == 0) ? threadsNeeded / grid.x : (threadsNeeded / grid.x) + 1;
        }
        // Swap orig/result list
        float4* tempList = d_origList;
        d_origList = d_resultList;
        d_resultList = tempList;
        {
            std::lock_guard<gloop::HostLoop::KernelLock> lock(hostLoop.kernelLock());
            cudaBindTexture(0, tex, d_origList, channelDesc, listsize * sizeof(float));
        }

#if 1
        hostLoop.launchWithSharedMemory<LoopType>(hostContext, dim3(135), grid, threads, 0, [] __device__ (gloop::DeviceLoop<LoopType>* loop, float4* result, int nrElems, int threadsPerDiv) {
            mergeSortPassInitialKernel(loop, result, nrElems, threadsPerDiv);
        }, d_resultList, nrElems, threadsPerDiv);
#endif

        nrElems *= 2;
        floatsperthread = (nrElems * 4);
        if (threadsPerDiv == 1)
            break;
    }
////////////////////////////////////////////////////////////////////////////
// Now, get rid of the NULL elements
////////////////////////////////////////////////////////////////////////////
#ifdef MERGE_WG_SIZE_0
    threads.x = MERGE_WG_SIZE_0;
#else
    threads.x = 256;
#endif
    grid.x = ((largestSize % threads.x) == 0) ? largestSize / threads.x : (largestSize / threads.x) + 1;
    grid.y = divisions;
    mergepack(hostLoop, hostContext, grid, threads, (float*)d_resultList, (float*)d_origList);

    free(startaddr);
    return d_origList;
}
