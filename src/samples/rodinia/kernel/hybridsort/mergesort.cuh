#ifndef __MERGESORT
#define __MERGESORT

#include "bucketsort.cuh"

extern texture<float4, 1, cudaReadModeElementType> tex;
extern texture<float, 1, cudaReadModeElementType> txt;
extern __constant__ int constStartAddr[DIVISIONS + 1];
extern __constant__ int finalStartAddr[DIVISIONS + 1];
extern __constant__ int nullElems[DIVISIONS];

void mergeSortFirst(gloop::HostLoop& hostLoop, gloop::HostContext& hostContext, dim3 grid, dim3 threads, float4* result, int listsize);
void mergepack(gloop::HostLoop& hostLoop, gloop::HostContext& hostContext, dim3 grid, dim3 threads, float* d_resultList, float* d_origList);
void mergeSortPass(gloop::HostLoop& hostLoop, gloop::HostContext& hostContext, dim3 grid, dim3 threads, float4* result, int nrElems, int threadsPerDiv);

float4* runMergeSort(gloop::HostLoop&, gloop::HostContext&, int listsize, int divisions,
    float4* d_origList, float4* d_resultList,
    int* sizes, int* nullElements,
    unsigned int* origOffsets);

#endif
