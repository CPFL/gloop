#ifndef __MERGESORT
#define __MERGESORT

#include "bucketsort.cuh"

float4* runMergeSort(gloop::HostLoop&, gloop::HostContext&, int listsize, int divisions,
    float4* d_origList, float4* d_resultList,
    int* sizes, int* nullElements,
    unsigned int* origOffsets);

#endif
