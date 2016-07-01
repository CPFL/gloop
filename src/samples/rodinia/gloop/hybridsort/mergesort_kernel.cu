#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>
#include "mergesort_inlines.cuh"

__device__ void mergeSortPassKernel(gloop::DeviceLoop<gloop::Global>* loop, float4* result, int nrElems, int threadsPerDiv)
{
    int tid = (loop->logicalBlockIdx().x * blockDim.x) + threadIdx.x;
    // The division to work on
    int division = tid / threadsPerDiv;
    if (division >= DIVISIONS)
        return;

    // The block within the division
    int int_tid = tid - division * threadsPerDiv;
    int Astart = constStartAddr[division] + int_tid * nrElems;

    int Bstart = Astart + nrElems / 2;
    float4* resStart = &(result[Astart]);

    if (Astart >= constStartAddr[division + 1])
        return;
    if (Bstart >= constStartAddr[division + 1]) {
        for (int i = 0; i < (constStartAddr[division + 1] - Astart); i++) {
            resStart[i] = tex1Dfetch(tex, Astart + i);
        }
        return;
    }

    int aidx = 0;
    int bidx = 0;
    int outidx = 0;
    float4 a, b;
    a = tex1Dfetch(tex, Astart + aidx);
    b = tex1Dfetch(tex, Bstart + bidx);

    while (true) //aidx < nrElems/2)// || (bidx < nrElems/2  && (Bstart + bidx < constEndAddr[division])))
    {
        /**
		 * For some reason, it's faster to do the texture fetches here than
		 * after the merge
		 */
        float4 nextA = tex1Dfetch(tex, Astart + aidx + 1);
        float4 nextB = tex1Dfetch(tex, Bstart + bidx + 1);

        float4 na = getLowest(a, b);
        float4 nb = getHighest(a, b);
        a = sortElem(na);
        b = sortElem(nb);
        // Now, a contains the lowest four elements, sorted
        resStart[outidx++] = a;

        bool elemsLeftInA;
        bool elemsLeftInB;

        elemsLeftInA = (aidx + 1 < nrElems / 2); // Astart + aidx + 1 is allways less than division border
        elemsLeftInB = (bidx + 1 < nrElems / 2) && (Bstart + bidx + 1 < constStartAddr[division + 1]);

        if (elemsLeftInA) {
            if (elemsLeftInB) {
                if (nextA.x < nextB.x) {
                    aidx += 1;
                    a = nextA;
                }
                else {
                    bidx += 1;
                    a = nextB;
                }
            }
            else {
                aidx += 1;
                a = nextA;
            }
        }
        else {
            if (elemsLeftInB) {
                bidx += 1;
                a = nextB;
            }
            else {
                break;
            }
        }
    }
    resStart[outidx++] = b;
}


#endif // #ifndef _MATRIXMUL_KERNEL_H_
