#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>
#include "mergesort_inlines.cuh"

struct MergeSortContext {
    float4* a;
    float4* b;
    int* aidx;
    int* bidx;
    int* disabled;
};

static __device__ void destroyContext(MergeSortContext* context)
{
    BEGIN_SINGLE_THREAD
    {
        delete [] context->a;
        delete [] context->b;
        delete [] context->aidx;
        delete [] context->bidx;
        delete [] context->disabled;
        delete context;
    }
    END_SINGLE_THREAD
}

static __device__ void mergeSortPassKernel(gloop::DeviceLoop<gloop::Shared>* loop, MergeSortContext* context, float4* result, int nrElems, int threadsPerDiv, int outidx, int Astart, int Bstart, int tid, int division, int aidx, int bidx, float4 a, float4 b, int disabled)
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

        if (gloop::loop::postTaskIfNecessary(loop, [context, result, nrElems, threadsPerDiv, outidx] (gloop::DeviceLoop<>* loop) {
            int tid = 0;
            int division = 0;
            int int_tid = 0;
            int Astart = 0;
            int Bstart = 0;
            int disabled = context->disabled[threadIdx.x];
            if (!disabled) {
                tid = (loop->logicalBlockIdx().x * blockDim.x) + threadIdx.x;
                division = tid / threadsPerDiv;
                int_tid = tid - division * threadsPerDiv;
                Astart = constStartAddr[division] + int_tid * nrElems;
                Bstart = Astart + nrElems / 2;
            }
            mergeSortPassKernel(loop, context, result, nrElems, threadsPerDiv, outidx, Astart, Bstart, tid, division, context->aidx[threadIdx.x], context->bidx[threadIdx.x], context->a[threadIdx.x], context->b[threadIdx.x], disabled);
        })) {
            context->aidx[threadIdx.x] = aidx;
            context->bidx[threadIdx.x] = bidx;
            context->a[threadIdx.x] = a;
            context->b[threadIdx.x] = b;
            context->disabled[threadIdx.x] = disabled;
            return;
        }
    }

    if (!disabled) {
        resStart[outidx++] = b;
    }
    destroyContext(context);
}

static __device__ void mergeSortPassInitialKernel(gloop::DeviceLoop<gloop::Shared>* loop, float4* result, int nrElems, int threadsPerDiv)
{
    __shared__ MergeSortContext* context;
    BEGIN_SINGLE_THREAD
    {
        context = new MergeSortContext;
        context->a = new float4[blockDim.x];
        GLOOP_ASSERT(context->a);
        context->b = new float4[blockDim.x];
        GLOOP_ASSERT(context->b);
        context->aidx = new int[blockDim.x];
        GLOOP_ASSERT(context->aidx);
        context->bidx = new int[blockDim.x];
        GLOOP_ASSERT(context->bidx);
        context->disabled = new int[blockDim.x];
        GLOOP_ASSERT(context->disabled);
    }
    END_SINGLE_THREAD

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


#endif // #ifndef _MATRIXMUL_KERNEL_H_
