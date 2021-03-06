#ifndef __MERGESORT
#define __MERGESORT

#include "bucketsort.cuh"

extern texture<float4, 1, cudaReadModeElementType> tex;
extern texture<float, 1, cudaReadModeElementType> txt;
extern __constant__ int constStartAddr[DIVISIONS + 1];
extern __constant__ int finalStartAddr[DIVISIONS + 1];
extern __constant__ int nullElems[DIVISIONS];

void mergeSortFirst(Context*, dim3 grid, dim3 threads, float4* result, int listsize);
void mergepack(Context*, dim3 grid, dim3 threads, float* d_resultList, float* d_origList);
void mergeSortPass(dim3 grid, dim3 threads, float4* result, int nrElems, int threadsPerDiv);

float4* runMergeSort(Context* ctx,
    int listsize, int divisions,
    float4* d_origList, float4* d_resultList,
    int* sizes, int* nullElements,
    unsigned int* origOffsets);

#endif
