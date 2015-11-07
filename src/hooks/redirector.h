/*
  Copyright (C) 2015 Yusuke Suzuki <yusuke.suzuki@sslab.ics.keio.ac.jp>

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
  ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
  THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#ifndef GNODE_HOOKS_REDIRECTOR_H_
#define GNODE_HOOKS_REDIRECTOR_H_
#include <cuda_runtime_api.h>
namespace gnode {
namespace hooks {

class Redirector {
public:
cudaError_t cudaDeviceReset();
cudaError_t cudaDeviceSynchronize();
cudaError_t cudaDeviceSetLimit(enum cudaLimit limit, size_t value);
cudaError_t cudaDeviceGetLimit(size_t * pValue, enum cudaLimit limit);
cudaError_t cudaDeviceGetCacheConfig(enum cudaFuncCache * pCacheConfig);
cudaError_t cudaDeviceSetCacheConfig(enum cudaFuncCache cacheConfig);
cudaError_t cudaDeviceGetSharedMemConfig(enum cudaSharedMemConfig * pConfig);
cudaError_t cudaDeviceSetSharedMemConfig(enum cudaSharedMemConfig config);
cudaError_t cudaDeviceGetByPCIBusId(int * device, char * pciBusId);
cudaError_t cudaDeviceGetPCIBusId(char * pciBusId, int len, int device);
cudaError_t cudaIpcGetEventHandle(cudaIpcEventHandle_t * handle, cudaEvent_t event);
cudaError_t cudaIpcOpenEventHandle(cudaEvent_t * event, cudaIpcEventHandle_t handle);
cudaError_t cudaIpcGetMemHandle(cudaIpcMemHandle_t * handle, void * devPtr);
cudaError_t cudaIpcOpenMemHandle(void ** devPtr, cudaIpcMemHandle_t handle, unsigned int flags);
cudaError_t cudaIpcCloseMemHandle(void * devPtr);
cudaError_t cudaThreadExit();
cudaError_t cudaThreadSynchronize();
cudaError_t cudaThreadSetLimit(enum cudaLimit limit, size_t value);
cudaError_t cudaThreadGetLimit(size_t * pValue, enum cudaLimit limit);
cudaError_t cudaThreadGetCacheConfig(enum cudaFuncCache * pCacheConfig);
cudaError_t cudaThreadSetCacheConfig(enum cudaFuncCache cacheConfig);
cudaError_t cudaGetLastError();
cudaError_t cudaPeekAtLastError();
const char * cudaGetErrorString(cudaError_t error);
cudaError_t cudaGetDeviceCount(int * count);
cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp * prop, int device);
cudaError_t cudaDeviceGetAttribute(int * value, enum cudaDeviceAttr attr, int device);
cudaError_t cudaChooseDevice(int * device, const struct cudaDeviceProp * prop);
cudaError_t cudaSetDevice(int device);
cudaError_t cudaGetDevice(int * device);
cudaError_t cudaSetValidDevices(int * device_arr, int len);
cudaError_t cudaSetDeviceFlags(unsigned int flags);
cudaError_t cudaStreamCreate(cudaStream_t * pStream);
cudaError_t cudaStreamCreateWithFlags(cudaStream_t * pStream, unsigned int flags);
cudaError_t cudaStreamDestroy(cudaStream_t stream);
cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags);
cudaError_t cudaStreamAddCallback(cudaStream_t stream, cudaStreamCallback_t callback, void * userData, unsigned int flags);
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
cudaError_t cudaStreamQuery(cudaStream_t stream);
cudaError_t cudaEventCreate(cudaEvent_t * event);
cudaError_t cudaEventCreateWithFlags(cudaEvent_t * event, unsigned int flags);
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream);
cudaError_t cudaEventQuery(cudaEvent_t event);
cudaError_t cudaEventSynchronize(cudaEvent_t event);
cudaError_t cudaEventDestroy(cudaEvent_t event);
cudaError_t cudaEventElapsedTime(float * ms, cudaEvent_t start, cudaEvent_t end);
cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream);
cudaError_t cudaSetupArgument(const void * arg, size_t size, size_t offset);
cudaError_t cudaFuncSetCacheConfig(const void * func, enum cudaFuncCache cacheConfig);
cudaError_t cudaFuncSetSharedMemConfig(const void * func, enum cudaSharedMemConfig config);
cudaError_t cudaLaunch(const void * func);
cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes * attr, const void * func);
cudaError_t cudaSetDoubleForDevice(double * d);
cudaError_t cudaSetDoubleForHost(double * d);
cudaError_t cudaMalloc(void ** devPtr, size_t size);
cudaError_t cudaMallocHost(void ** ptr, size_t size);
cudaError_t cudaMallocPitch(void ** devPtr, size_t * pitch, size_t width, size_t height);
cudaError_t cudaMallocArray(cudaArray_t * array, const struct cudaChannelFormatDesc * desc, size_t width, size_t height, unsigned int flags);
cudaError_t cudaFree(void * devPtr);
cudaError_t cudaFreeHost(void * ptr);
cudaError_t cudaFreeArray(cudaArray_t array);
cudaError_t cudaFreeMipmappedArray(cudaMipmappedArray_t mipmappedArray);
cudaError_t cudaHostAlloc(void ** pHost, size_t size, unsigned int flags);
cudaError_t cudaHostRegister(void * ptr, size_t size, unsigned int flags);
cudaError_t cudaHostUnregister(void * ptr);
cudaError_t cudaHostGetDevicePointer(void ** pDevice, void * pHost, unsigned int flags);
cudaError_t cudaHostGetFlags(unsigned int * pFlags, void * pHost);
cudaError_t cudaMalloc3D(struct cudaPitchedPtr * pitchedDevPtr, struct cudaExtent extent);
cudaError_t cudaMalloc3DArray(cudaArray_t * array, const struct cudaChannelFormatDesc * desc, struct cudaExtent extent, unsigned int flags);
cudaError_t cudaMallocMipmappedArray(cudaMipmappedArray_t * mipmappedArray, const struct cudaChannelFormatDesc * desc, struct cudaExtent extent, unsigned int numLevels, unsigned int flags);
cudaError_t cudaGetMipmappedArrayLevel(cudaArray_t * levelArray, cudaMipmappedArray_const_t mipmappedArray, unsigned int level);
cudaError_t cudaMemcpy3D(const struct cudaMemcpy3DParms * p);
cudaError_t cudaMemcpy3DPeer(const struct cudaMemcpy3DPeerParms * p);
cudaError_t cudaMemcpy3DAsync(const struct cudaMemcpy3DParms * p, cudaStream_t stream);
cudaError_t cudaMemcpy3DPeerAsync(const struct cudaMemcpy3DPeerParms * p, cudaStream_t stream);
cudaError_t cudaMemGetInfo(size_t * free, size_t * total);
cudaError_t cudaArrayGetInfo(struct cudaChannelFormatDesc * desc, struct cudaExtent * extent, unsigned int * flags, cudaArray_t array);
cudaError_t cudaMemcpy(void * dst, const void * src, size_t count, enum cudaMemcpyKind kind);
cudaError_t cudaMemcpyPeer(void * dst, int dstDevice, const void * src, int srcDevice, size_t count);
cudaError_t cudaMemcpyToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t count, enum cudaMemcpyKind kind);
cudaError_t cudaMemcpyFromArray(void * dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind);
cudaError_t cudaMemcpyArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, enum cudaMemcpyKind kind);
cudaError_t cudaMemcpy2D(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind);
cudaError_t cudaMemcpy2DToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind);
cudaError_t cudaMemcpy2DFromArray(void * dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind);
cudaError_t cudaMemcpy2DArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, enum cudaMemcpyKind kind);
cudaError_t cudaMemcpyToSymbol(const void * symbol, const void * src, size_t count, size_t offset, enum cudaMemcpyKind kind);
cudaError_t cudaMemcpyFromSymbol(void * dst, const void * symbol, size_t count, size_t offset, enum cudaMemcpyKind kind);
cudaError_t cudaMemcpyAsync(void * dst, const void * src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t cudaMemcpyPeerAsync(void * dst, int dstDevice, const void * src, int srcDevice, size_t count, cudaStream_t stream);
cudaError_t cudaMemcpyToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t cudaMemcpyFromArrayAsync(void * dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t cudaMemcpy2DAsync(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t cudaMemcpy2DToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t cudaMemcpy2DFromArrayAsync(void * dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t cudaMemcpyToSymbolAsync(const void * symbol, const void * src, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t cudaMemcpyFromSymbolAsync(void * dst, const void * symbol, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t cudaMemset(void * devPtr, int value, size_t count);
cudaError_t cudaMemset2D(void * devPtr, size_t pitch, int value, size_t width, size_t height);
cudaError_t cudaMemset3D(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent);
cudaError_t cudaMemsetAsync(void * devPtr, int value, size_t count, cudaStream_t stream);
cudaError_t cudaMemset2DAsync(void * devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream);
cudaError_t cudaMemset3DAsync(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent, cudaStream_t stream);
cudaError_t cudaGetSymbolAddress(void ** devPtr, const void * symbol);
cudaError_t cudaGetSymbolSize(size_t * size, const void * symbol);
cudaError_t cudaPointerGetAttributes(struct cudaPointerAttributes * attributes, const void * ptr);
cudaError_t cudaDeviceCanAccessPeer(int * canAccessPeer, int device, int peerDevice);
cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags);
cudaError_t cudaDeviceDisablePeerAccess(int peerDevice);
cudaError_t cudaGraphicsUnregisterResource(cudaGraphicsResource_t resource);
cudaError_t cudaGraphicsResourceSetMapFlags(cudaGraphicsResource_t resource, unsigned int flags);
cudaError_t cudaGraphicsMapResources(int count, cudaGraphicsResource_t * resources, cudaStream_t stream);
cudaError_t cudaGraphicsUnmapResources(int count, cudaGraphicsResource_t * resources, cudaStream_t stream);
cudaError_t cudaGraphicsResourceGetMappedPointer(void ** devPtr, size_t * size, cudaGraphicsResource_t resource);
cudaError_t cudaGraphicsSubResourceGetMappedArray(cudaArray_t * array, cudaGraphicsResource_t resource, unsigned int arrayIndex, unsigned int mipLevel);
cudaError_t cudaGraphicsResourceGetMappedMipmappedArray(cudaMipmappedArray_t * mipmappedArray, cudaGraphicsResource_t resource);
cudaError_t cudaGetChannelDesc(struct cudaChannelFormatDesc * desc, cudaArray_const_t array);
struct cudaChannelFormatDesc cudaCreateChannelDesc(int x, int y, int z, int w, enum cudaChannelFormatKind f);
cudaError_t cudaBindTexture(size_t * offset, const struct textureReference * texref, const void * devPtr, const struct cudaChannelFormatDesc * desc, size_t size);
cudaError_t cudaBindTexture2D(size_t * offset, const struct textureReference * texref, const void * devPtr, const struct cudaChannelFormatDesc * desc, size_t width, size_t height, size_t pitch);
cudaError_t cudaBindTextureToArray(const struct textureReference * texref, cudaArray_const_t array, const struct cudaChannelFormatDesc * desc);
cudaError_t cudaBindTextureToMipmappedArray(const struct textureReference * texref, cudaMipmappedArray_const_t mipmappedArray, const struct cudaChannelFormatDesc * desc);
cudaError_t cudaUnbindTexture(const struct textureReference * texref);
cudaError_t cudaGetTextureAlignmentOffset(size_t * offset, const struct textureReference * texref);
cudaError_t cudaGetTextureReference(const struct textureReference ** texref, const void * symbol);
cudaError_t cudaBindSurfaceToArray(const struct surfaceReference * surfref, cudaArray_const_t array, const struct cudaChannelFormatDesc * desc);
cudaError_t cudaGetSurfaceReference(const struct surfaceReference ** surfref, const void * symbol);
cudaError_t cudaCreateTextureObject(cudaTextureObject_t * pTexObject, const struct cudaResourceDesc * pResDesc, const struct cudaTextureDesc * pTexDesc, const struct cudaResourceViewDesc * pResViewDesc);
cudaError_t cudaDestroyTextureObject(cudaTextureObject_t texObject);
cudaError_t cudaGetTextureObjectResourceDesc(struct cudaResourceDesc * pResDesc, cudaTextureObject_t texObject);
cudaError_t cudaGetTextureObjectTextureDesc(struct cudaTextureDesc * pTexDesc, cudaTextureObject_t texObject);
cudaError_t cudaGetTextureObjectResourceViewDesc(struct cudaResourceViewDesc * pResViewDesc, cudaTextureObject_t texObject);
cudaError_t cudaCreateSurfaceObject(cudaSurfaceObject_t * pSurfObject, const struct cudaResourceDesc * pResDesc);
cudaError_t cudaDestroySurfaceObject(cudaSurfaceObject_t surfObject);
cudaError_t cudaGetSurfaceObjectResourceDesc(struct cudaResourceDesc * pResDesc, cudaSurfaceObject_t surfObject);
cudaError_t cudaDriverGetVersion(int * driverVersion);
cudaError_t cudaRuntimeGetVersion(int * runtimeVersion);
cudaError_t cudaGetExportTable(const void ** ppExportTable, const cudaUUID_t * pExportTableId);
private:
typedef cudaError_t (*APIcudaDeviceReset)();
APIcudaDeviceReset m_cudaDeviceReset;

typedef cudaError_t (*APIcudaDeviceSynchronize)();
APIcudaDeviceSynchronize m_cudaDeviceSynchronize;

typedef cudaError_t (*APIcudaDeviceSetLimit)(enum cudaLimit, size_t);
APIcudaDeviceSetLimit m_cudaDeviceSetLimit;

typedef cudaError_t (*APIcudaDeviceGetLimit)(size_t *, enum cudaLimit);
APIcudaDeviceGetLimit m_cudaDeviceGetLimit;

typedef cudaError_t (*APIcudaDeviceGetCacheConfig)(enum cudaFuncCache *);
APIcudaDeviceGetCacheConfig m_cudaDeviceGetCacheConfig;

typedef cudaError_t (*APIcudaDeviceSetCacheConfig)(enum cudaFuncCache);
APIcudaDeviceSetCacheConfig m_cudaDeviceSetCacheConfig;

typedef cudaError_t (*APIcudaDeviceGetSharedMemConfig)(enum cudaSharedMemConfig *);
APIcudaDeviceGetSharedMemConfig m_cudaDeviceGetSharedMemConfig;

typedef cudaError_t (*APIcudaDeviceSetSharedMemConfig)(enum cudaSharedMemConfig);
APIcudaDeviceSetSharedMemConfig m_cudaDeviceSetSharedMemConfig;

typedef cudaError_t (*APIcudaDeviceGetByPCIBusId)(int *, char *);
APIcudaDeviceGetByPCIBusId m_cudaDeviceGetByPCIBusId;

typedef cudaError_t (*APIcudaDeviceGetPCIBusId)(char *, int, int);
APIcudaDeviceGetPCIBusId m_cudaDeviceGetPCIBusId;

typedef cudaError_t (*APIcudaIpcGetEventHandle)(cudaIpcEventHandle_t *, cudaEvent_t);
APIcudaIpcGetEventHandle m_cudaIpcGetEventHandle;

typedef cudaError_t (*APIcudaIpcOpenEventHandle)(cudaEvent_t *, cudaIpcEventHandle_t);
APIcudaIpcOpenEventHandle m_cudaIpcOpenEventHandle;

typedef cudaError_t (*APIcudaIpcGetMemHandle)(cudaIpcMemHandle_t *, void *);
APIcudaIpcGetMemHandle m_cudaIpcGetMemHandle;

typedef cudaError_t (*APIcudaIpcOpenMemHandle)(void **, cudaIpcMemHandle_t, unsigned int);
APIcudaIpcOpenMemHandle m_cudaIpcOpenMemHandle;

typedef cudaError_t (*APIcudaIpcCloseMemHandle)(void *);
APIcudaIpcCloseMemHandle m_cudaIpcCloseMemHandle;

typedef cudaError_t (*APIcudaThreadExit)();
APIcudaThreadExit m_cudaThreadExit;

typedef cudaError_t (*APIcudaThreadSynchronize)();
APIcudaThreadSynchronize m_cudaThreadSynchronize;

typedef cudaError_t (*APIcudaThreadSetLimit)(enum cudaLimit, size_t);
APIcudaThreadSetLimit m_cudaThreadSetLimit;

typedef cudaError_t (*APIcudaThreadGetLimit)(size_t *, enum cudaLimit);
APIcudaThreadGetLimit m_cudaThreadGetLimit;

typedef cudaError_t (*APIcudaThreadGetCacheConfig)(enum cudaFuncCache *);
APIcudaThreadGetCacheConfig m_cudaThreadGetCacheConfig;

typedef cudaError_t (*APIcudaThreadSetCacheConfig)(enum cudaFuncCache);
APIcudaThreadSetCacheConfig m_cudaThreadSetCacheConfig;

typedef cudaError_t (*APIcudaGetLastError)();
APIcudaGetLastError m_cudaGetLastError;

typedef cudaError_t (*APIcudaPeekAtLastError)();
APIcudaPeekAtLastError m_cudaPeekAtLastError;

typedef const char * (*APIcudaGetErrorString)(cudaError_t);
APIcudaGetErrorString m_cudaGetErrorString;

typedef cudaError_t (*APIcudaGetDeviceCount)(int *);
APIcudaGetDeviceCount m_cudaGetDeviceCount;

typedef cudaError_t (*APIcudaGetDeviceProperties)(struct cudaDeviceProp *, int);
APIcudaGetDeviceProperties m_cudaGetDeviceProperties;

typedef cudaError_t (*APIcudaDeviceGetAttribute)(int *, enum cudaDeviceAttr, int);
APIcudaDeviceGetAttribute m_cudaDeviceGetAttribute;

typedef cudaError_t (*APIcudaChooseDevice)(int *, const struct cudaDeviceProp *);
APIcudaChooseDevice m_cudaChooseDevice;

typedef cudaError_t (*APIcudaSetDevice)(int);
APIcudaSetDevice m_cudaSetDevice;

typedef cudaError_t (*APIcudaGetDevice)(int *);
APIcudaGetDevice m_cudaGetDevice;

typedef cudaError_t (*APIcudaSetValidDevices)(int *, int);
APIcudaSetValidDevices m_cudaSetValidDevices;

typedef cudaError_t (*APIcudaSetDeviceFlags)(unsigned int);
APIcudaSetDeviceFlags m_cudaSetDeviceFlags;

typedef cudaError_t (*APIcudaStreamCreate)(cudaStream_t *);
APIcudaStreamCreate m_cudaStreamCreate;

typedef cudaError_t (*APIcudaStreamCreateWithFlags)(cudaStream_t *, unsigned int);
APIcudaStreamCreateWithFlags m_cudaStreamCreateWithFlags;

typedef cudaError_t (*APIcudaStreamDestroy)(cudaStream_t);
APIcudaStreamDestroy m_cudaStreamDestroy;

typedef cudaError_t (*APIcudaStreamWaitEvent)(cudaStream_t, cudaEvent_t, unsigned int);
APIcudaStreamWaitEvent m_cudaStreamWaitEvent;

typedef cudaError_t (*APIcudaStreamAddCallback)(cudaStream_t, cudaStreamCallback_t, void *, unsigned int);
APIcudaStreamAddCallback m_cudaStreamAddCallback;

typedef cudaError_t (*APIcudaStreamSynchronize)(cudaStream_t);
APIcudaStreamSynchronize m_cudaStreamSynchronize;

typedef cudaError_t (*APIcudaStreamQuery)(cudaStream_t);
APIcudaStreamQuery m_cudaStreamQuery;

typedef cudaError_t (*APIcudaEventCreate)(cudaEvent_t *);
APIcudaEventCreate m_cudaEventCreate;

typedef cudaError_t (*APIcudaEventCreateWithFlags)(cudaEvent_t *, unsigned int);
APIcudaEventCreateWithFlags m_cudaEventCreateWithFlags;

typedef cudaError_t (*APIcudaEventRecord)(cudaEvent_t, cudaStream_t);
APIcudaEventRecord m_cudaEventRecord;

typedef cudaError_t (*APIcudaEventQuery)(cudaEvent_t);
APIcudaEventQuery m_cudaEventQuery;

typedef cudaError_t (*APIcudaEventSynchronize)(cudaEvent_t);
APIcudaEventSynchronize m_cudaEventSynchronize;

typedef cudaError_t (*APIcudaEventDestroy)(cudaEvent_t);
APIcudaEventDestroy m_cudaEventDestroy;

typedef cudaError_t (*APIcudaEventElapsedTime)(float *, cudaEvent_t, cudaEvent_t);
APIcudaEventElapsedTime m_cudaEventElapsedTime;

typedef cudaError_t (*APIcudaConfigureCall)(dim3, dim3, size_t, cudaStream_t);
APIcudaConfigureCall m_cudaConfigureCall;

typedef cudaError_t (*APIcudaSetupArgument)(const void *, size_t, size_t);
APIcudaSetupArgument m_cudaSetupArgument;

typedef cudaError_t (*APIcudaFuncSetCacheConfig)(const void *, enum cudaFuncCache);
APIcudaFuncSetCacheConfig m_cudaFuncSetCacheConfig;

typedef cudaError_t (*APIcudaFuncSetSharedMemConfig)(const void *, enum cudaSharedMemConfig);
APIcudaFuncSetSharedMemConfig m_cudaFuncSetSharedMemConfig;

typedef cudaError_t (*APIcudaLaunch)(const void *);
APIcudaLaunch m_cudaLaunch;

typedef cudaError_t (*APIcudaFuncGetAttributes)(struct cudaFuncAttributes *, const void *);
APIcudaFuncGetAttributes m_cudaFuncGetAttributes;

typedef cudaError_t (*APIcudaSetDoubleForDevice)(double *);
APIcudaSetDoubleForDevice m_cudaSetDoubleForDevice;

typedef cudaError_t (*APIcudaSetDoubleForHost)(double *);
APIcudaSetDoubleForHost m_cudaSetDoubleForHost;

typedef cudaError_t (*APIcudaMalloc)(void **, size_t);
APIcudaMalloc m_cudaMalloc;

typedef cudaError_t (*APIcudaMallocHost)(void **, size_t);
APIcudaMallocHost m_cudaMallocHost;

typedef cudaError_t (*APIcudaMallocPitch)(void **, size_t *, size_t, size_t);
APIcudaMallocPitch m_cudaMallocPitch;

typedef cudaError_t (*APIcudaMallocArray)(cudaArray_t *, const struct cudaChannelFormatDesc *, size_t, size_t, unsigned int);
APIcudaMallocArray m_cudaMallocArray;

typedef cudaError_t (*APIcudaFree)(void *);
APIcudaFree m_cudaFree;

typedef cudaError_t (*APIcudaFreeHost)(void *);
APIcudaFreeHost m_cudaFreeHost;

typedef cudaError_t (*APIcudaFreeArray)(cudaArray_t);
APIcudaFreeArray m_cudaFreeArray;

typedef cudaError_t (*APIcudaFreeMipmappedArray)(cudaMipmappedArray_t);
APIcudaFreeMipmappedArray m_cudaFreeMipmappedArray;

typedef cudaError_t (*APIcudaHostAlloc)(void **, size_t, unsigned int);
APIcudaHostAlloc m_cudaHostAlloc;

typedef cudaError_t (*APIcudaHostRegister)(void *, size_t, unsigned int);
APIcudaHostRegister m_cudaHostRegister;

typedef cudaError_t (*APIcudaHostUnregister)(void *);
APIcudaHostUnregister m_cudaHostUnregister;

typedef cudaError_t (*APIcudaHostGetDevicePointer)(void **, void *, unsigned int);
APIcudaHostGetDevicePointer m_cudaHostGetDevicePointer;

typedef cudaError_t (*APIcudaHostGetFlags)(unsigned int *, void *);
APIcudaHostGetFlags m_cudaHostGetFlags;

typedef cudaError_t (*APIcudaMalloc3D)(struct cudaPitchedPtr *, struct cudaExtent);
APIcudaMalloc3D m_cudaMalloc3D;

typedef cudaError_t (*APIcudaMalloc3DArray)(cudaArray_t *, const struct cudaChannelFormatDesc *, struct cudaExtent, unsigned int);
APIcudaMalloc3DArray m_cudaMalloc3DArray;

typedef cudaError_t (*APIcudaMallocMipmappedArray)(cudaMipmappedArray_t *, const struct cudaChannelFormatDesc *, struct cudaExtent, unsigned int, unsigned int);
APIcudaMallocMipmappedArray m_cudaMallocMipmappedArray;

typedef cudaError_t (*APIcudaGetMipmappedArrayLevel)(cudaArray_t *, cudaMipmappedArray_const_t, unsigned int);
APIcudaGetMipmappedArrayLevel m_cudaGetMipmappedArrayLevel;

typedef cudaError_t (*APIcudaMemcpy3D)(const struct cudaMemcpy3DParms *);
APIcudaMemcpy3D m_cudaMemcpy3D;

typedef cudaError_t (*APIcudaMemcpy3DPeer)(const struct cudaMemcpy3DPeerParms *);
APIcudaMemcpy3DPeer m_cudaMemcpy3DPeer;

typedef cudaError_t (*APIcudaMemcpy3DAsync)(const struct cudaMemcpy3DParms *, cudaStream_t);
APIcudaMemcpy3DAsync m_cudaMemcpy3DAsync;

typedef cudaError_t (*APIcudaMemcpy3DPeerAsync)(const struct cudaMemcpy3DPeerParms *, cudaStream_t);
APIcudaMemcpy3DPeerAsync m_cudaMemcpy3DPeerAsync;

typedef cudaError_t (*APIcudaMemGetInfo)(size_t *, size_t *);
APIcudaMemGetInfo m_cudaMemGetInfo;

typedef cudaError_t (*APIcudaArrayGetInfo)(struct cudaChannelFormatDesc *, struct cudaExtent *, unsigned int *, cudaArray_t);
APIcudaArrayGetInfo m_cudaArrayGetInfo;

typedef cudaError_t (*APIcudaMemcpy)(void *, const void *, size_t, enum cudaMemcpyKind);
APIcudaMemcpy m_cudaMemcpy;

typedef cudaError_t (*APIcudaMemcpyPeer)(void *, int, const void *, int, size_t);
APIcudaMemcpyPeer m_cudaMemcpyPeer;

typedef cudaError_t (*APIcudaMemcpyToArray)(cudaArray_t, size_t, size_t, const void *, size_t, enum cudaMemcpyKind);
APIcudaMemcpyToArray m_cudaMemcpyToArray;

typedef cudaError_t (*APIcudaMemcpyFromArray)(void *, cudaArray_const_t, size_t, size_t, size_t, enum cudaMemcpyKind);
APIcudaMemcpyFromArray m_cudaMemcpyFromArray;

typedef cudaError_t (*APIcudaMemcpyArrayToArray)(cudaArray_t, size_t, size_t, cudaArray_const_t, size_t, size_t, size_t, enum cudaMemcpyKind);
APIcudaMemcpyArrayToArray m_cudaMemcpyArrayToArray;

typedef cudaError_t (*APIcudaMemcpy2D)(void *, size_t, const void *, size_t, size_t, size_t, enum cudaMemcpyKind);
APIcudaMemcpy2D m_cudaMemcpy2D;

typedef cudaError_t (*APIcudaMemcpy2DToArray)(cudaArray_t, size_t, size_t, const void *, size_t, size_t, size_t, enum cudaMemcpyKind);
APIcudaMemcpy2DToArray m_cudaMemcpy2DToArray;

typedef cudaError_t (*APIcudaMemcpy2DFromArray)(void *, size_t, cudaArray_const_t, size_t, size_t, size_t, size_t, enum cudaMemcpyKind);
APIcudaMemcpy2DFromArray m_cudaMemcpy2DFromArray;

typedef cudaError_t (*APIcudaMemcpy2DArrayToArray)(cudaArray_t, size_t, size_t, cudaArray_const_t, size_t, size_t, size_t, size_t, enum cudaMemcpyKind);
APIcudaMemcpy2DArrayToArray m_cudaMemcpy2DArrayToArray;

typedef cudaError_t (*APIcudaMemcpyToSymbol)(const void *, const void *, size_t, size_t, enum cudaMemcpyKind);
APIcudaMemcpyToSymbol m_cudaMemcpyToSymbol;

typedef cudaError_t (*APIcudaMemcpyFromSymbol)(void *, const void *, size_t, size_t, enum cudaMemcpyKind);
APIcudaMemcpyFromSymbol m_cudaMemcpyFromSymbol;

typedef cudaError_t (*APIcudaMemcpyAsync)(void *, const void *, size_t, enum cudaMemcpyKind, cudaStream_t);
APIcudaMemcpyAsync m_cudaMemcpyAsync;

typedef cudaError_t (*APIcudaMemcpyPeerAsync)(void *, int, const void *, int, size_t, cudaStream_t);
APIcudaMemcpyPeerAsync m_cudaMemcpyPeerAsync;

typedef cudaError_t (*APIcudaMemcpyToArrayAsync)(cudaArray_t, size_t, size_t, const void *, size_t, enum cudaMemcpyKind, cudaStream_t);
APIcudaMemcpyToArrayAsync m_cudaMemcpyToArrayAsync;

typedef cudaError_t (*APIcudaMemcpyFromArrayAsync)(void *, cudaArray_const_t, size_t, size_t, size_t, enum cudaMemcpyKind, cudaStream_t);
APIcudaMemcpyFromArrayAsync m_cudaMemcpyFromArrayAsync;

typedef cudaError_t (*APIcudaMemcpy2DAsync)(void *, size_t, const void *, size_t, size_t, size_t, enum cudaMemcpyKind, cudaStream_t);
APIcudaMemcpy2DAsync m_cudaMemcpy2DAsync;

typedef cudaError_t (*APIcudaMemcpy2DToArrayAsync)(cudaArray_t, size_t, size_t, const void *, size_t, size_t, size_t, enum cudaMemcpyKind, cudaStream_t);
APIcudaMemcpy2DToArrayAsync m_cudaMemcpy2DToArrayAsync;

typedef cudaError_t (*APIcudaMemcpy2DFromArrayAsync)(void *, size_t, cudaArray_const_t, size_t, size_t, size_t, size_t, enum cudaMemcpyKind, cudaStream_t);
APIcudaMemcpy2DFromArrayAsync m_cudaMemcpy2DFromArrayAsync;

typedef cudaError_t (*APIcudaMemcpyToSymbolAsync)(const void *, const void *, size_t, size_t, enum cudaMemcpyKind, cudaStream_t);
APIcudaMemcpyToSymbolAsync m_cudaMemcpyToSymbolAsync;

typedef cudaError_t (*APIcudaMemcpyFromSymbolAsync)(void *, const void *, size_t, size_t, enum cudaMemcpyKind, cudaStream_t);
APIcudaMemcpyFromSymbolAsync m_cudaMemcpyFromSymbolAsync;

typedef cudaError_t (*APIcudaMemset)(void *, int, size_t);
APIcudaMemset m_cudaMemset;

typedef cudaError_t (*APIcudaMemset2D)(void *, size_t, int, size_t, size_t);
APIcudaMemset2D m_cudaMemset2D;

typedef cudaError_t (*APIcudaMemset3D)(struct cudaPitchedPtr, int, struct cudaExtent);
APIcudaMemset3D m_cudaMemset3D;

typedef cudaError_t (*APIcudaMemsetAsync)(void *, int, size_t, cudaStream_t);
APIcudaMemsetAsync m_cudaMemsetAsync;

typedef cudaError_t (*APIcudaMemset2DAsync)(void *, size_t, int, size_t, size_t, cudaStream_t);
APIcudaMemset2DAsync m_cudaMemset2DAsync;

typedef cudaError_t (*APIcudaMemset3DAsync)(struct cudaPitchedPtr, int, struct cudaExtent, cudaStream_t);
APIcudaMemset3DAsync m_cudaMemset3DAsync;

typedef cudaError_t (*APIcudaGetSymbolAddress)(void **, const void *);
APIcudaGetSymbolAddress m_cudaGetSymbolAddress;

typedef cudaError_t (*APIcudaGetSymbolSize)(size_t *, const void *);
APIcudaGetSymbolSize m_cudaGetSymbolSize;

typedef cudaError_t (*APIcudaPointerGetAttributes)(struct cudaPointerAttributes *, const void *);
APIcudaPointerGetAttributes m_cudaPointerGetAttributes;

typedef cudaError_t (*APIcudaDeviceCanAccessPeer)(int *, int, int);
APIcudaDeviceCanAccessPeer m_cudaDeviceCanAccessPeer;

typedef cudaError_t (*APIcudaDeviceEnablePeerAccess)(int, unsigned int);
APIcudaDeviceEnablePeerAccess m_cudaDeviceEnablePeerAccess;

typedef cudaError_t (*APIcudaDeviceDisablePeerAccess)(int);
APIcudaDeviceDisablePeerAccess m_cudaDeviceDisablePeerAccess;

typedef cudaError_t (*APIcudaGraphicsUnregisterResource)(cudaGraphicsResource_t);
APIcudaGraphicsUnregisterResource m_cudaGraphicsUnregisterResource;

typedef cudaError_t (*APIcudaGraphicsResourceSetMapFlags)(cudaGraphicsResource_t, unsigned int);
APIcudaGraphicsResourceSetMapFlags m_cudaGraphicsResourceSetMapFlags;

typedef cudaError_t (*APIcudaGraphicsMapResources)(int, cudaGraphicsResource_t *, cudaStream_t);
APIcudaGraphicsMapResources m_cudaGraphicsMapResources;

typedef cudaError_t (*APIcudaGraphicsUnmapResources)(int, cudaGraphicsResource_t *, cudaStream_t);
APIcudaGraphicsUnmapResources m_cudaGraphicsUnmapResources;

typedef cudaError_t (*APIcudaGraphicsResourceGetMappedPointer)(void **, size_t *, cudaGraphicsResource_t);
APIcudaGraphicsResourceGetMappedPointer m_cudaGraphicsResourceGetMappedPointer;

typedef cudaError_t (*APIcudaGraphicsSubResourceGetMappedArray)(cudaArray_t *, cudaGraphicsResource_t, unsigned int, unsigned int);
APIcudaGraphicsSubResourceGetMappedArray m_cudaGraphicsSubResourceGetMappedArray;

typedef cudaError_t (*APIcudaGraphicsResourceGetMappedMipmappedArray)(cudaMipmappedArray_t *, cudaGraphicsResource_t);
APIcudaGraphicsResourceGetMappedMipmappedArray m_cudaGraphicsResourceGetMappedMipmappedArray;

typedef cudaError_t (*APIcudaGetChannelDesc)(struct cudaChannelFormatDesc *, cudaArray_const_t);
APIcudaGetChannelDesc m_cudaGetChannelDesc;

typedef struct cudaChannelFormatDesc (*APIcudaCreateChannelDesc)(int, int, int, int, enum cudaChannelFormatKind);
APIcudaCreateChannelDesc m_cudaCreateChannelDesc;

typedef cudaError_t (*APIcudaBindTexture)(size_t *, const struct textureReference *, const void *, const struct cudaChannelFormatDesc *, size_t);
APIcudaBindTexture m_cudaBindTexture;

typedef cudaError_t (*APIcudaBindTexture2D)(size_t *, const struct textureReference *, const void *, const struct cudaChannelFormatDesc *, size_t, size_t, size_t);
APIcudaBindTexture2D m_cudaBindTexture2D;

typedef cudaError_t (*APIcudaBindTextureToArray)(const struct textureReference *, cudaArray_const_t, const struct cudaChannelFormatDesc *);
APIcudaBindTextureToArray m_cudaBindTextureToArray;

typedef cudaError_t (*APIcudaBindTextureToMipmappedArray)(const struct textureReference *, cudaMipmappedArray_const_t, const struct cudaChannelFormatDesc *);
APIcudaBindTextureToMipmappedArray m_cudaBindTextureToMipmappedArray;

typedef cudaError_t (*APIcudaUnbindTexture)(const struct textureReference *);
APIcudaUnbindTexture m_cudaUnbindTexture;

typedef cudaError_t (*APIcudaGetTextureAlignmentOffset)(size_t *, const struct textureReference *);
APIcudaGetTextureAlignmentOffset m_cudaGetTextureAlignmentOffset;

typedef cudaError_t (*APIcudaGetTextureReference)(const struct textureReference **, const void *);
APIcudaGetTextureReference m_cudaGetTextureReference;

typedef cudaError_t (*APIcudaBindSurfaceToArray)(const struct surfaceReference *, cudaArray_const_t, const struct cudaChannelFormatDesc *);
APIcudaBindSurfaceToArray m_cudaBindSurfaceToArray;

typedef cudaError_t (*APIcudaGetSurfaceReference)(const struct surfaceReference **, const void *);
APIcudaGetSurfaceReference m_cudaGetSurfaceReference;

typedef cudaError_t (*APIcudaCreateTextureObject)(cudaTextureObject_t *, const struct cudaResourceDesc *, const struct cudaTextureDesc *, const struct cudaResourceViewDesc *);
APIcudaCreateTextureObject m_cudaCreateTextureObject;

typedef cudaError_t (*APIcudaDestroyTextureObject)(cudaTextureObject_t);
APIcudaDestroyTextureObject m_cudaDestroyTextureObject;

typedef cudaError_t (*APIcudaGetTextureObjectResourceDesc)(struct cudaResourceDesc *, cudaTextureObject_t);
APIcudaGetTextureObjectResourceDesc m_cudaGetTextureObjectResourceDesc;

typedef cudaError_t (*APIcudaGetTextureObjectTextureDesc)(struct cudaTextureDesc *, cudaTextureObject_t);
APIcudaGetTextureObjectTextureDesc m_cudaGetTextureObjectTextureDesc;

typedef cudaError_t (*APIcudaGetTextureObjectResourceViewDesc)(struct cudaResourceViewDesc *, cudaTextureObject_t);
APIcudaGetTextureObjectResourceViewDesc m_cudaGetTextureObjectResourceViewDesc;

typedef cudaError_t (*APIcudaCreateSurfaceObject)(cudaSurfaceObject_t *, const struct cudaResourceDesc *);
APIcudaCreateSurfaceObject m_cudaCreateSurfaceObject;

typedef cudaError_t (*APIcudaDestroySurfaceObject)(cudaSurfaceObject_t);
APIcudaDestroySurfaceObject m_cudaDestroySurfaceObject;

typedef cudaError_t (*APIcudaGetSurfaceObjectResourceDesc)(struct cudaResourceDesc *, cudaSurfaceObject_t);
APIcudaGetSurfaceObjectResourceDesc m_cudaGetSurfaceObjectResourceDesc;

typedef cudaError_t (*APIcudaDriverGetVersion)(int *);
APIcudaDriverGetVersion m_cudaDriverGetVersion;

typedef cudaError_t (*APIcudaRuntimeGetVersion)(int *);
APIcudaRuntimeGetVersion m_cudaRuntimeGetVersion;

typedef cudaError_t (*APIcudaGetExportTable)(const void **, const cudaUUID_t *);
APIcudaGetExportTable m_cudaGetExportTable;

protected:
Redirector();

};
} }  // namespace gnode::hooks

extern "C" {
cudaError_t cudaDeviceReset();
cudaError_t cudaDeviceSynchronize();
cudaError_t cudaDeviceSetLimit(enum cudaLimit limit, size_t value);
cudaError_t cudaDeviceGetLimit(size_t * pValue, enum cudaLimit limit);
cudaError_t cudaDeviceGetCacheConfig(enum cudaFuncCache * pCacheConfig);
cudaError_t cudaDeviceSetCacheConfig(enum cudaFuncCache cacheConfig);
cudaError_t cudaDeviceGetSharedMemConfig(enum cudaSharedMemConfig * pConfig);
cudaError_t cudaDeviceSetSharedMemConfig(enum cudaSharedMemConfig config);
cudaError_t cudaDeviceGetByPCIBusId(int * device, char * pciBusId);
cudaError_t cudaDeviceGetPCIBusId(char * pciBusId, int len, int device);
cudaError_t cudaIpcGetEventHandle(cudaIpcEventHandle_t * handle, cudaEvent_t event);
cudaError_t cudaIpcOpenEventHandle(cudaEvent_t * event, cudaIpcEventHandle_t handle);
cudaError_t cudaIpcGetMemHandle(cudaIpcMemHandle_t * handle, void * devPtr);
cudaError_t cudaIpcOpenMemHandle(void ** devPtr, cudaIpcMemHandle_t handle, unsigned int flags);
cudaError_t cudaIpcCloseMemHandle(void * devPtr);
cudaError_t cudaThreadExit();
cudaError_t cudaThreadSynchronize();
cudaError_t cudaThreadSetLimit(enum cudaLimit limit, size_t value);
cudaError_t cudaThreadGetLimit(size_t * pValue, enum cudaLimit limit);
cudaError_t cudaThreadGetCacheConfig(enum cudaFuncCache * pCacheConfig);
cudaError_t cudaThreadSetCacheConfig(enum cudaFuncCache cacheConfig);
cudaError_t cudaGetLastError();
cudaError_t cudaPeekAtLastError();
const char * cudaGetErrorString(cudaError_t error);
cudaError_t cudaGetDeviceCount(int * count);
cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp * prop, int device);
cudaError_t cudaDeviceGetAttribute(int * value, enum cudaDeviceAttr attr, int device);
cudaError_t cudaChooseDevice(int * device, const struct cudaDeviceProp * prop);
cudaError_t cudaSetDevice(int device);
cudaError_t cudaGetDevice(int * device);
cudaError_t cudaSetValidDevices(int * device_arr, int len);
cudaError_t cudaSetDeviceFlags(unsigned int flags);
cudaError_t cudaStreamCreate(cudaStream_t * pStream);
cudaError_t cudaStreamCreateWithFlags(cudaStream_t * pStream, unsigned int flags);
cudaError_t cudaStreamDestroy(cudaStream_t stream);
cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags);
cudaError_t cudaStreamAddCallback(cudaStream_t stream, cudaStreamCallback_t callback, void * userData, unsigned int flags);
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
cudaError_t cudaStreamQuery(cudaStream_t stream);
cudaError_t cudaEventCreate(cudaEvent_t * event);
cudaError_t cudaEventCreateWithFlags(cudaEvent_t * event, unsigned int flags);
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream);
cudaError_t cudaEventQuery(cudaEvent_t event);
cudaError_t cudaEventSynchronize(cudaEvent_t event);
cudaError_t cudaEventDestroy(cudaEvent_t event);
cudaError_t cudaEventElapsedTime(float * ms, cudaEvent_t start, cudaEvent_t end);
cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream);
cudaError_t cudaSetupArgument(const void * arg, size_t size, size_t offset);
cudaError_t cudaFuncSetCacheConfig(const void * func, enum cudaFuncCache cacheConfig);
cudaError_t cudaFuncSetSharedMemConfig(const void * func, enum cudaSharedMemConfig config);
cudaError_t cudaLaunch(const void * func);
cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes * attr, const void * func);
cudaError_t cudaSetDoubleForDevice(double * d);
cudaError_t cudaSetDoubleForHost(double * d);
cudaError_t cudaMalloc(void ** devPtr, size_t size);
cudaError_t cudaMallocHost(void ** ptr, size_t size);
cudaError_t cudaMallocPitch(void ** devPtr, size_t * pitch, size_t width, size_t height);
cudaError_t cudaMallocArray(cudaArray_t * array, const struct cudaChannelFormatDesc * desc, size_t width, size_t height, unsigned int flags);
cudaError_t cudaFree(void * devPtr);
cudaError_t cudaFreeHost(void * ptr);
cudaError_t cudaFreeArray(cudaArray_t array);
cudaError_t cudaFreeMipmappedArray(cudaMipmappedArray_t mipmappedArray);
cudaError_t cudaHostAlloc(void ** pHost, size_t size, unsigned int flags);
cudaError_t cudaHostRegister(void * ptr, size_t size, unsigned int flags);
cudaError_t cudaHostUnregister(void * ptr);
cudaError_t cudaHostGetDevicePointer(void ** pDevice, void * pHost, unsigned int flags);
cudaError_t cudaHostGetFlags(unsigned int * pFlags, void * pHost);
cudaError_t cudaMalloc3D(struct cudaPitchedPtr * pitchedDevPtr, struct cudaExtent extent);
cudaError_t cudaMalloc3DArray(cudaArray_t * array, const struct cudaChannelFormatDesc * desc, struct cudaExtent extent, unsigned int flags);
cudaError_t cudaMallocMipmappedArray(cudaMipmappedArray_t * mipmappedArray, const struct cudaChannelFormatDesc * desc, struct cudaExtent extent, unsigned int numLevels, unsigned int flags);
cudaError_t cudaGetMipmappedArrayLevel(cudaArray_t * levelArray, cudaMipmappedArray_const_t mipmappedArray, unsigned int level);
cudaError_t cudaMemcpy3D(const struct cudaMemcpy3DParms * p);
cudaError_t cudaMemcpy3DPeer(const struct cudaMemcpy3DPeerParms * p);
cudaError_t cudaMemcpy3DAsync(const struct cudaMemcpy3DParms * p, cudaStream_t stream);
cudaError_t cudaMemcpy3DPeerAsync(const struct cudaMemcpy3DPeerParms * p, cudaStream_t stream);
cudaError_t cudaMemGetInfo(size_t * free, size_t * total);
cudaError_t cudaArrayGetInfo(struct cudaChannelFormatDesc * desc, struct cudaExtent * extent, unsigned int * flags, cudaArray_t array);
cudaError_t cudaMemcpy(void * dst, const void * src, size_t count, enum cudaMemcpyKind kind);
cudaError_t cudaMemcpyPeer(void * dst, int dstDevice, const void * src, int srcDevice, size_t count);
cudaError_t cudaMemcpyToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t count, enum cudaMemcpyKind kind);
cudaError_t cudaMemcpyFromArray(void * dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind);
cudaError_t cudaMemcpyArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, enum cudaMemcpyKind kind);
cudaError_t cudaMemcpy2D(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind);
cudaError_t cudaMemcpy2DToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind);
cudaError_t cudaMemcpy2DFromArray(void * dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind);
cudaError_t cudaMemcpy2DArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, enum cudaMemcpyKind kind);
cudaError_t cudaMemcpyToSymbol(const void * symbol, const void * src, size_t count, size_t offset, enum cudaMemcpyKind kind);
cudaError_t cudaMemcpyFromSymbol(void * dst, const void * symbol, size_t count, size_t offset, enum cudaMemcpyKind kind);
cudaError_t cudaMemcpyAsync(void * dst, const void * src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t cudaMemcpyPeerAsync(void * dst, int dstDevice, const void * src, int srcDevice, size_t count, cudaStream_t stream);
cudaError_t cudaMemcpyToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t cudaMemcpyFromArrayAsync(void * dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t cudaMemcpy2DAsync(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t cudaMemcpy2DToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t cudaMemcpy2DFromArrayAsync(void * dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t cudaMemcpyToSymbolAsync(const void * symbol, const void * src, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t cudaMemcpyFromSymbolAsync(void * dst, const void * symbol, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t cudaMemset(void * devPtr, int value, size_t count);
cudaError_t cudaMemset2D(void * devPtr, size_t pitch, int value, size_t width, size_t height);
cudaError_t cudaMemset3D(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent);
cudaError_t cudaMemsetAsync(void * devPtr, int value, size_t count, cudaStream_t stream);
cudaError_t cudaMemset2DAsync(void * devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream);
cudaError_t cudaMemset3DAsync(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent, cudaStream_t stream);
cudaError_t cudaGetSymbolAddress(void ** devPtr, const void * symbol);
cudaError_t cudaGetSymbolSize(size_t * size, const void * symbol);
cudaError_t cudaPointerGetAttributes(struct cudaPointerAttributes * attributes, const void * ptr);
cudaError_t cudaDeviceCanAccessPeer(int * canAccessPeer, int device, int peerDevice);
cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags);
cudaError_t cudaDeviceDisablePeerAccess(int peerDevice);
cudaError_t cudaGraphicsUnregisterResource(cudaGraphicsResource_t resource);
cudaError_t cudaGraphicsResourceSetMapFlags(cudaGraphicsResource_t resource, unsigned int flags);
cudaError_t cudaGraphicsMapResources(int count, cudaGraphicsResource_t * resources, cudaStream_t stream);
cudaError_t cudaGraphicsUnmapResources(int count, cudaGraphicsResource_t * resources, cudaStream_t stream);
cudaError_t cudaGraphicsResourceGetMappedPointer(void ** devPtr, size_t * size, cudaGraphicsResource_t resource);
cudaError_t cudaGraphicsSubResourceGetMappedArray(cudaArray_t * array, cudaGraphicsResource_t resource, unsigned int arrayIndex, unsigned int mipLevel);
cudaError_t cudaGraphicsResourceGetMappedMipmappedArray(cudaMipmappedArray_t * mipmappedArray, cudaGraphicsResource_t resource);
cudaError_t cudaGetChannelDesc(struct cudaChannelFormatDesc * desc, cudaArray_const_t array);
struct cudaChannelFormatDesc cudaCreateChannelDesc(int x, int y, int z, int w, enum cudaChannelFormatKind f);
cudaError_t cudaBindTexture(size_t * offset, const struct textureReference * texref, const void * devPtr, const struct cudaChannelFormatDesc * desc, size_t size);
cudaError_t cudaBindTexture2D(size_t * offset, const struct textureReference * texref, const void * devPtr, const struct cudaChannelFormatDesc * desc, size_t width, size_t height, size_t pitch);
cudaError_t cudaBindTextureToArray(const struct textureReference * texref, cudaArray_const_t array, const struct cudaChannelFormatDesc * desc);
cudaError_t cudaBindTextureToMipmappedArray(const struct textureReference * texref, cudaMipmappedArray_const_t mipmappedArray, const struct cudaChannelFormatDesc * desc);
cudaError_t cudaUnbindTexture(const struct textureReference * texref);
cudaError_t cudaGetTextureAlignmentOffset(size_t * offset, const struct textureReference * texref);
cudaError_t cudaGetTextureReference(const struct textureReference ** texref, const void * symbol);
cudaError_t cudaBindSurfaceToArray(const struct surfaceReference * surfref, cudaArray_const_t array, const struct cudaChannelFormatDesc * desc);
cudaError_t cudaGetSurfaceReference(const struct surfaceReference ** surfref, const void * symbol);
cudaError_t cudaCreateTextureObject(cudaTextureObject_t * pTexObject, const struct cudaResourceDesc * pResDesc, const struct cudaTextureDesc * pTexDesc, const struct cudaResourceViewDesc * pResViewDesc);
cudaError_t cudaDestroyTextureObject(cudaTextureObject_t texObject);
cudaError_t cudaGetTextureObjectResourceDesc(struct cudaResourceDesc * pResDesc, cudaTextureObject_t texObject);
cudaError_t cudaGetTextureObjectTextureDesc(struct cudaTextureDesc * pTexDesc, cudaTextureObject_t texObject);
cudaError_t cudaGetTextureObjectResourceViewDesc(struct cudaResourceViewDesc * pResViewDesc, cudaTextureObject_t texObject);
cudaError_t cudaCreateSurfaceObject(cudaSurfaceObject_t * pSurfObject, const struct cudaResourceDesc * pResDesc);
cudaError_t cudaDestroySurfaceObject(cudaSurfaceObject_t surfObject);
cudaError_t cudaGetSurfaceObjectResourceDesc(struct cudaResourceDesc * pResDesc, cudaSurfaceObject_t surfObject);
cudaError_t cudaDriverGetVersion(int * driverVersion);
cudaError_t cudaRuntimeGetVersion(int * runtimeVersion);
cudaError_t cudaGetExportTable(const void ** ppExportTable, const cudaUUID_t * pExportTableId);
}
#endif  // GNODE_HOOKS_REDIRECTOR_H_

