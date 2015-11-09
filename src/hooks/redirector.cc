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
#include "redirector.h"
#include "main_loop.h"
#include <cuda_runtime_api.h>
#include <dlfcn.h>
namespace gnode {
namespace hooks {


cudaError_t Redirector::cudaDeviceReset()
{
    return this->m_cudaDeviceReset();
}

cudaError_t Redirector::cudaDeviceSynchronize()
{
    return this->m_cudaDeviceSynchronize();
}

cudaError_t Redirector::cudaDeviceSetLimit(enum cudaLimit limit, size_t value)
{
    return this->m_cudaDeviceSetLimit(limit, value);
}

cudaError_t Redirector::cudaDeviceGetLimit(size_t * pValue, enum cudaLimit limit)
{
    return this->m_cudaDeviceGetLimit(pValue, limit);
}

cudaError_t Redirector::cudaDeviceGetCacheConfig(enum cudaFuncCache * pCacheConfig)
{
    return this->m_cudaDeviceGetCacheConfig(pCacheConfig);
}

cudaError_t Redirector::cudaDeviceSetCacheConfig(enum cudaFuncCache cacheConfig)
{
    return this->m_cudaDeviceSetCacheConfig(cacheConfig);
}

cudaError_t Redirector::cudaDeviceGetSharedMemConfig(enum cudaSharedMemConfig * pConfig)
{
    return this->m_cudaDeviceGetSharedMemConfig(pConfig);
}

cudaError_t Redirector::cudaDeviceSetSharedMemConfig(enum cudaSharedMemConfig config)
{
    return this->m_cudaDeviceSetSharedMemConfig(config);
}

cudaError_t Redirector::cudaDeviceGetByPCIBusId(int * device, char * pciBusId)
{
    return this->m_cudaDeviceGetByPCIBusId(device, pciBusId);
}

cudaError_t Redirector::cudaDeviceGetPCIBusId(char * pciBusId, int len, int device)
{
    return this->m_cudaDeviceGetPCIBusId(pciBusId, len, device);
}

cudaError_t Redirector::cudaIpcGetEventHandle(cudaIpcEventHandle_t * handle, cudaEvent_t event)
{
    return this->m_cudaIpcGetEventHandle(handle, event);
}

cudaError_t Redirector::cudaIpcOpenEventHandle(cudaEvent_t * event, cudaIpcEventHandle_t handle)
{
    return this->m_cudaIpcOpenEventHandle(event, handle);
}

cudaError_t Redirector::cudaIpcGetMemHandle(cudaIpcMemHandle_t * handle, void * devPtr)
{
    return this->m_cudaIpcGetMemHandle(handle, devPtr);
}

cudaError_t Redirector::cudaIpcOpenMemHandle(void ** devPtr, cudaIpcMemHandle_t handle, unsigned int flags)
{
    return this->m_cudaIpcOpenMemHandle(devPtr, handle, flags);
}

cudaError_t Redirector::cudaIpcCloseMemHandle(void * devPtr)
{
    return this->m_cudaIpcCloseMemHandle(devPtr);
}

cudaError_t Redirector::cudaThreadExit()
{
    return this->m_cudaThreadExit();
}

cudaError_t Redirector::cudaThreadSynchronize()
{
    return this->m_cudaThreadSynchronize();
}

cudaError_t Redirector::cudaThreadSetLimit(enum cudaLimit limit, size_t value)
{
    return this->m_cudaThreadSetLimit(limit, value);
}

cudaError_t Redirector::cudaThreadGetLimit(size_t * pValue, enum cudaLimit limit)
{
    return this->m_cudaThreadGetLimit(pValue, limit);
}

cudaError_t Redirector::cudaThreadGetCacheConfig(enum cudaFuncCache * pCacheConfig)
{
    return this->m_cudaThreadGetCacheConfig(pCacheConfig);
}

cudaError_t Redirector::cudaThreadSetCacheConfig(enum cudaFuncCache cacheConfig)
{
    return this->m_cudaThreadSetCacheConfig(cacheConfig);
}

cudaError_t Redirector::cudaGetLastError()
{
    return this->m_cudaGetLastError();
}

cudaError_t Redirector::cudaPeekAtLastError()
{
    return this->m_cudaPeekAtLastError();
}

const char * Redirector::cudaGetErrorString(cudaError_t error)
{
    return this->m_cudaGetErrorString(error);
}

cudaError_t Redirector::cudaGetDeviceCount(int * count)
{
    return this->m_cudaGetDeviceCount(count);
}

cudaError_t Redirector::cudaGetDeviceProperties(struct cudaDeviceProp * prop, int device)
{
    return this->m_cudaGetDeviceProperties(prop, device);
}

cudaError_t Redirector::cudaDeviceGetAttribute(int * value, enum cudaDeviceAttr attr, int device)
{
    return this->m_cudaDeviceGetAttribute(value, attr, device);
}

cudaError_t Redirector::cudaChooseDevice(int * device, const struct cudaDeviceProp * prop)
{
    return this->m_cudaChooseDevice(device, prop);
}

cudaError_t Redirector::cudaSetDevice(int device)
{
    return this->m_cudaSetDevice(device);
}

cudaError_t Redirector::cudaGetDevice(int * device)
{
    return this->m_cudaGetDevice(device);
}

cudaError_t Redirector::cudaSetValidDevices(int * device_arr, int len)
{
    return this->m_cudaSetValidDevices(device_arr, len);
}

cudaError_t Redirector::cudaSetDeviceFlags(unsigned int flags)
{
    return this->m_cudaSetDeviceFlags(flags);
}

cudaError_t Redirector::cudaStreamCreate(cudaStream_t * pStream)
{
    return this->m_cudaStreamCreate(pStream);
}

cudaError_t Redirector::cudaStreamCreateWithFlags(cudaStream_t * pStream, unsigned int flags)
{
    return this->m_cudaStreamCreateWithFlags(pStream, flags);
}

cudaError_t Redirector::cudaStreamDestroy(cudaStream_t stream)
{
    return this->m_cudaStreamDestroy(stream);
}

cudaError_t Redirector::cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags)
{
    return this->m_cudaStreamWaitEvent(stream, event, flags);
}

cudaError_t Redirector::cudaStreamAddCallback(cudaStream_t stream, cudaStreamCallback_t callback, void * userData, unsigned int flags)
{
    return this->m_cudaStreamAddCallback(stream, callback, userData, flags);
}

cudaError_t Redirector::cudaStreamSynchronize(cudaStream_t stream)
{
    return this->m_cudaStreamSynchronize(stream);
}

cudaError_t Redirector::cudaStreamQuery(cudaStream_t stream)
{
    return this->m_cudaStreamQuery(stream);
}

cudaError_t Redirector::cudaEventCreate(cudaEvent_t * event)
{
    return this->m_cudaEventCreate(event);
}

cudaError_t Redirector::cudaEventCreateWithFlags(cudaEvent_t * event, unsigned int flags)
{
    return this->m_cudaEventCreateWithFlags(event, flags);
}

cudaError_t Redirector::cudaEventRecord(cudaEvent_t event, cudaStream_t stream)
{
    return this->m_cudaEventRecord(event, stream);
}

cudaError_t Redirector::cudaEventQuery(cudaEvent_t event)
{
    return this->m_cudaEventQuery(event);
}

cudaError_t Redirector::cudaEventSynchronize(cudaEvent_t event)
{
    return this->m_cudaEventSynchronize(event);
}

cudaError_t Redirector::cudaEventDestroy(cudaEvent_t event)
{
    return this->m_cudaEventDestroy(event);
}

cudaError_t Redirector::cudaEventElapsedTime(float * ms, cudaEvent_t start, cudaEvent_t end)
{
    return this->m_cudaEventElapsedTime(ms, start, end);
}

cudaError_t Redirector::cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream)
{
    return this->m_cudaConfigureCall(gridDim, blockDim, sharedMem, stream);
}

cudaError_t Redirector::cudaSetupArgument(const void * arg, size_t size, size_t offset)
{
    return this->m_cudaSetupArgument(arg, size, offset);
}

cudaError_t Redirector::cudaFuncSetCacheConfig(const void * func, enum cudaFuncCache cacheConfig)
{
    return this->m_cudaFuncSetCacheConfig(func, cacheConfig);
}

cudaError_t Redirector::cudaFuncSetSharedMemConfig(const void * func, enum cudaSharedMemConfig config)
{
    return this->m_cudaFuncSetSharedMemConfig(func, config);
}

cudaError_t Redirector::cudaLaunch(const void * func)
{
    return this->m_cudaLaunch(func);
}

cudaError_t Redirector::cudaFuncGetAttributes(struct cudaFuncAttributes * attr, const void * func)
{
    return this->m_cudaFuncGetAttributes(attr, func);
}

cudaError_t Redirector::cudaSetDoubleForDevice(double * d)
{
    return this->m_cudaSetDoubleForDevice(d);
}

cudaError_t Redirector::cudaSetDoubleForHost(double * d)
{
    return this->m_cudaSetDoubleForHost(d);
}

cudaError_t Redirector::cudaMalloc(void ** devPtr, size_t size)
{
    return this->m_cudaMalloc(devPtr, size);
}

cudaError_t Redirector::cudaMallocHost(void ** ptr, size_t size)
{
    return this->m_cudaMallocHost(ptr, size);
}

cudaError_t Redirector::cudaMallocPitch(void ** devPtr, size_t * pitch, size_t width, size_t height)
{
    return this->m_cudaMallocPitch(devPtr, pitch, width, height);
}

cudaError_t Redirector::cudaMallocArray(cudaArray_t * array, const struct cudaChannelFormatDesc * desc, size_t width, size_t height, unsigned int flags)
{
    return this->m_cudaMallocArray(array, desc, width, height, flags);
}

cudaError_t Redirector::cudaFree(void * devPtr)
{
    return this->m_cudaFree(devPtr);
}

cudaError_t Redirector::cudaFreeHost(void * ptr)
{
    return this->m_cudaFreeHost(ptr);
}

cudaError_t Redirector::cudaFreeArray(cudaArray_t array)
{
    return this->m_cudaFreeArray(array);
}

cudaError_t Redirector::cudaFreeMipmappedArray(cudaMipmappedArray_t mipmappedArray)
{
    return this->m_cudaFreeMipmappedArray(mipmappedArray);
}

cudaError_t Redirector::cudaHostAlloc(void ** pHost, size_t size, unsigned int flags)
{
    return this->m_cudaHostAlloc(pHost, size, flags);
}

cudaError_t Redirector::cudaHostRegister(void * ptr, size_t size, unsigned int flags)
{
    return this->m_cudaHostRegister(ptr, size, flags);
}

cudaError_t Redirector::cudaHostUnregister(void * ptr)
{
    return this->m_cudaHostUnregister(ptr);
}

cudaError_t Redirector::cudaHostGetDevicePointer(void ** pDevice, void * pHost, unsigned int flags)
{
    return this->m_cudaHostGetDevicePointer(pDevice, pHost, flags);
}

cudaError_t Redirector::cudaHostGetFlags(unsigned int * pFlags, void * pHost)
{
    return this->m_cudaHostGetFlags(pFlags, pHost);
}

cudaError_t Redirector::cudaMalloc3D(struct cudaPitchedPtr * pitchedDevPtr, struct cudaExtent extent)
{
    return this->m_cudaMalloc3D(pitchedDevPtr, extent);
}

cudaError_t Redirector::cudaMalloc3DArray(cudaArray_t * array, const struct cudaChannelFormatDesc * desc, struct cudaExtent extent, unsigned int flags)
{
    return this->m_cudaMalloc3DArray(array, desc, extent, flags);
}

cudaError_t Redirector::cudaMallocMipmappedArray(cudaMipmappedArray_t * mipmappedArray, const struct cudaChannelFormatDesc * desc, struct cudaExtent extent, unsigned int numLevels, unsigned int flags)
{
    return this->m_cudaMallocMipmappedArray(mipmappedArray, desc, extent, numLevels, flags);
}

cudaError_t Redirector::cudaGetMipmappedArrayLevel(cudaArray_t * levelArray, cudaMipmappedArray_const_t mipmappedArray, unsigned int level)
{
    return this->m_cudaGetMipmappedArrayLevel(levelArray, mipmappedArray, level);
}

cudaError_t Redirector::cudaMemcpy3D(const struct cudaMemcpy3DParms * p)
{
    return this->m_cudaMemcpy3D(p);
}

cudaError_t Redirector::cudaMemcpy3DPeer(const struct cudaMemcpy3DPeerParms * p)
{
    return this->m_cudaMemcpy3DPeer(p);
}

cudaError_t Redirector::cudaMemcpy3DAsync(const struct cudaMemcpy3DParms * p, cudaStream_t stream)
{
    return this->m_cudaMemcpy3DAsync(p, stream);
}

cudaError_t Redirector::cudaMemcpy3DPeerAsync(const struct cudaMemcpy3DPeerParms * p, cudaStream_t stream)
{
    return this->m_cudaMemcpy3DPeerAsync(p, stream);
}

cudaError_t Redirector::cudaMemGetInfo(size_t * free, size_t * total)
{
    return this->m_cudaMemGetInfo(free, total);
}

cudaError_t Redirector::cudaArrayGetInfo(struct cudaChannelFormatDesc * desc, struct cudaExtent * extent, unsigned int * flags, cudaArray_t array)
{
    return this->m_cudaArrayGetInfo(desc, extent, flags, array);
}

cudaError_t Redirector::cudaMemcpy(void * dst, const void * src, size_t count, enum cudaMemcpyKind kind)
{
    return this->m_cudaMemcpy(dst, src, count, kind);
}

cudaError_t Redirector::cudaMemcpyPeer(void * dst, int dstDevice, const void * src, int srcDevice, size_t count)
{
    return this->m_cudaMemcpyPeer(dst, dstDevice, src, srcDevice, count);
}

cudaError_t Redirector::cudaMemcpyToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t count, enum cudaMemcpyKind kind)
{
    return this->m_cudaMemcpyToArray(dst, wOffset, hOffset, src, count, kind);
}

cudaError_t Redirector::cudaMemcpyFromArray(void * dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind)
{
    return this->m_cudaMemcpyFromArray(dst, src, wOffset, hOffset, count, kind);
}

cudaError_t Redirector::cudaMemcpyArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, enum cudaMemcpyKind kind)
{
    return this->m_cudaMemcpyArrayToArray(dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc, count, kind);
}

cudaError_t Redirector::cudaMemcpy2D(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind)
{
    return this->m_cudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind);
}

cudaError_t Redirector::cudaMemcpy2DToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind)
{
    return this->m_cudaMemcpy2DToArray(dst, wOffset, hOffset, src, spitch, width, height, kind);
}

cudaError_t Redirector::cudaMemcpy2DFromArray(void * dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind)
{
    return this->m_cudaMemcpy2DFromArray(dst, dpitch, src, wOffset, hOffset, width, height, kind);
}

cudaError_t Redirector::cudaMemcpy2DArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, enum cudaMemcpyKind kind)
{
    return this->m_cudaMemcpy2DArrayToArray(dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc, width, height, kind);
}

cudaError_t Redirector::cudaMemcpyToSymbol(const void * symbol, const void * src, size_t count, size_t offset, enum cudaMemcpyKind kind)
{
    return this->m_cudaMemcpyToSymbol(symbol, src, count, offset, kind);
}

cudaError_t Redirector::cudaMemcpyFromSymbol(void * dst, const void * symbol, size_t count, size_t offset, enum cudaMemcpyKind kind)
{
    return this->m_cudaMemcpyFromSymbol(dst, symbol, count, offset, kind);
}

cudaError_t Redirector::cudaMemcpyAsync(void * dst, const void * src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream)
{
    return this->m_cudaMemcpyAsync(dst, src, count, kind, stream);
}

cudaError_t Redirector::cudaMemcpyPeerAsync(void * dst, int dstDevice, const void * src, int srcDevice, size_t count, cudaStream_t stream)
{
    return this->m_cudaMemcpyPeerAsync(dst, dstDevice, src, srcDevice, count, stream);
}

cudaError_t Redirector::cudaMemcpyToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream)
{
    return this->m_cudaMemcpyToArrayAsync(dst, wOffset, hOffset, src, count, kind, stream);
}

cudaError_t Redirector::cudaMemcpyFromArrayAsync(void * dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream)
{
    return this->m_cudaMemcpyFromArrayAsync(dst, src, wOffset, hOffset, count, kind, stream);
}

cudaError_t Redirector::cudaMemcpy2DAsync(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream)
{
    return this->m_cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, stream);
}

cudaError_t Redirector::cudaMemcpy2DToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream)
{
    return this->m_cudaMemcpy2DToArrayAsync(dst, wOffset, hOffset, src, spitch, width, height, kind, stream);
}

cudaError_t Redirector::cudaMemcpy2DFromArrayAsync(void * dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream)
{
    return this->m_cudaMemcpy2DFromArrayAsync(dst, dpitch, src, wOffset, hOffset, width, height, kind, stream);
}

cudaError_t Redirector::cudaMemcpyToSymbolAsync(const void * symbol, const void * src, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream)
{
    return this->m_cudaMemcpyToSymbolAsync(symbol, src, count, offset, kind, stream);
}

cudaError_t Redirector::cudaMemcpyFromSymbolAsync(void * dst, const void * symbol, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream)
{
    return this->m_cudaMemcpyFromSymbolAsync(dst, symbol, count, offset, kind, stream);
}

cudaError_t Redirector::cudaMemset(void * devPtr, int value, size_t count)
{
    return this->m_cudaMemset(devPtr, value, count);
}

cudaError_t Redirector::cudaMemset2D(void * devPtr, size_t pitch, int value, size_t width, size_t height)
{
    return this->m_cudaMemset2D(devPtr, pitch, value, width, height);
}

cudaError_t Redirector::cudaMemset3D(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent)
{
    return this->m_cudaMemset3D(pitchedDevPtr, value, extent);
}

cudaError_t Redirector::cudaMemsetAsync(void * devPtr, int value, size_t count, cudaStream_t stream)
{
    return this->m_cudaMemsetAsync(devPtr, value, count, stream);
}

cudaError_t Redirector::cudaMemset2DAsync(void * devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream)
{
    return this->m_cudaMemset2DAsync(devPtr, pitch, value, width, height, stream);
}

cudaError_t Redirector::cudaMemset3DAsync(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent, cudaStream_t stream)
{
    return this->m_cudaMemset3DAsync(pitchedDevPtr, value, extent, stream);
}

cudaError_t Redirector::cudaGetSymbolAddress(void ** devPtr, const void * symbol)
{
    return this->m_cudaGetSymbolAddress(devPtr, symbol);
}

cudaError_t Redirector::cudaGetSymbolSize(size_t * size, const void * symbol)
{
    return this->m_cudaGetSymbolSize(size, symbol);
}

cudaError_t Redirector::cudaPointerGetAttributes(struct cudaPointerAttributes * attributes, const void * ptr)
{
    return this->m_cudaPointerGetAttributes(attributes, ptr);
}

cudaError_t Redirector::cudaDeviceCanAccessPeer(int * canAccessPeer, int device, int peerDevice)
{
    return this->m_cudaDeviceCanAccessPeer(canAccessPeer, device, peerDevice);
}

cudaError_t Redirector::cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags)
{
    return this->m_cudaDeviceEnablePeerAccess(peerDevice, flags);
}

cudaError_t Redirector::cudaDeviceDisablePeerAccess(int peerDevice)
{
    return this->m_cudaDeviceDisablePeerAccess(peerDevice);
}

cudaError_t Redirector::cudaGraphicsUnregisterResource(cudaGraphicsResource_t resource)
{
    return this->m_cudaGraphicsUnregisterResource(resource);
}

cudaError_t Redirector::cudaGraphicsResourceSetMapFlags(cudaGraphicsResource_t resource, unsigned int flags)
{
    return this->m_cudaGraphicsResourceSetMapFlags(resource, flags);
}

cudaError_t Redirector::cudaGraphicsMapResources(int count, cudaGraphicsResource_t * resources, cudaStream_t stream)
{
    return this->m_cudaGraphicsMapResources(count, resources, stream);
}

cudaError_t Redirector::cudaGraphicsUnmapResources(int count, cudaGraphicsResource_t * resources, cudaStream_t stream)
{
    return this->m_cudaGraphicsUnmapResources(count, resources, stream);
}

cudaError_t Redirector::cudaGraphicsResourceGetMappedPointer(void ** devPtr, size_t * size, cudaGraphicsResource_t resource)
{
    return this->m_cudaGraphicsResourceGetMappedPointer(devPtr, size, resource);
}

cudaError_t Redirector::cudaGraphicsSubResourceGetMappedArray(cudaArray_t * array, cudaGraphicsResource_t resource, unsigned int arrayIndex, unsigned int mipLevel)
{
    return this->m_cudaGraphicsSubResourceGetMappedArray(array, resource, arrayIndex, mipLevel);
}

cudaError_t Redirector::cudaGraphicsResourceGetMappedMipmappedArray(cudaMipmappedArray_t * mipmappedArray, cudaGraphicsResource_t resource)
{
    return this->m_cudaGraphicsResourceGetMappedMipmappedArray(mipmappedArray, resource);
}

cudaError_t Redirector::cudaGetChannelDesc(struct cudaChannelFormatDesc * desc, cudaArray_const_t array)
{
    return this->m_cudaGetChannelDesc(desc, array);
}

struct cudaChannelFormatDesc Redirector::cudaCreateChannelDesc(int x, int y, int z, int w, enum cudaChannelFormatKind f)
{
    return this->m_cudaCreateChannelDesc(x, y, z, w, f);
}

cudaError_t Redirector::cudaBindTexture(size_t * offset, const struct textureReference * texref, const void * devPtr, const struct cudaChannelFormatDesc * desc, size_t size)
{
    return this->m_cudaBindTexture(offset, texref, devPtr, desc, size);
}

cudaError_t Redirector::cudaBindTexture2D(size_t * offset, const struct textureReference * texref, const void * devPtr, const struct cudaChannelFormatDesc * desc, size_t width, size_t height, size_t pitch)
{
    return this->m_cudaBindTexture2D(offset, texref, devPtr, desc, width, height, pitch);
}

cudaError_t Redirector::cudaBindTextureToArray(const struct textureReference * texref, cudaArray_const_t array, const struct cudaChannelFormatDesc * desc)
{
    return this->m_cudaBindTextureToArray(texref, array, desc);
}

cudaError_t Redirector::cudaBindTextureToMipmappedArray(const struct textureReference * texref, cudaMipmappedArray_const_t mipmappedArray, const struct cudaChannelFormatDesc * desc)
{
    return this->m_cudaBindTextureToMipmappedArray(texref, mipmappedArray, desc);
}

cudaError_t Redirector::cudaUnbindTexture(const struct textureReference * texref)
{
    return this->m_cudaUnbindTexture(texref);
}

cudaError_t Redirector::cudaGetTextureAlignmentOffset(size_t * offset, const struct textureReference * texref)
{
    return this->m_cudaGetTextureAlignmentOffset(offset, texref);
}

cudaError_t Redirector::cudaGetTextureReference(const struct textureReference ** texref, const void * symbol)
{
    return this->m_cudaGetTextureReference(texref, symbol);
}

cudaError_t Redirector::cudaBindSurfaceToArray(const struct surfaceReference * surfref, cudaArray_const_t array, const struct cudaChannelFormatDesc * desc)
{
    return this->m_cudaBindSurfaceToArray(surfref, array, desc);
}

cudaError_t Redirector::cudaGetSurfaceReference(const struct surfaceReference ** surfref, const void * symbol)
{
    return this->m_cudaGetSurfaceReference(surfref, symbol);
}

cudaError_t Redirector::cudaCreateTextureObject(cudaTextureObject_t * pTexObject, const struct cudaResourceDesc * pResDesc, const struct cudaTextureDesc * pTexDesc, const struct cudaResourceViewDesc * pResViewDesc)
{
    return this->m_cudaCreateTextureObject(pTexObject, pResDesc, pTexDesc, pResViewDesc);
}

cudaError_t Redirector::cudaDestroyTextureObject(cudaTextureObject_t texObject)
{
    return this->m_cudaDestroyTextureObject(texObject);
}

cudaError_t Redirector::cudaGetTextureObjectResourceDesc(struct cudaResourceDesc * pResDesc, cudaTextureObject_t texObject)
{
    return this->m_cudaGetTextureObjectResourceDesc(pResDesc, texObject);
}

cudaError_t Redirector::cudaGetTextureObjectTextureDesc(struct cudaTextureDesc * pTexDesc, cudaTextureObject_t texObject)
{
    return this->m_cudaGetTextureObjectTextureDesc(pTexDesc, texObject);
}

cudaError_t Redirector::cudaGetTextureObjectResourceViewDesc(struct cudaResourceViewDesc * pResViewDesc, cudaTextureObject_t texObject)
{
    return this->m_cudaGetTextureObjectResourceViewDesc(pResViewDesc, texObject);
}

cudaError_t Redirector::cudaCreateSurfaceObject(cudaSurfaceObject_t * pSurfObject, const struct cudaResourceDesc * pResDesc)
{
    return this->m_cudaCreateSurfaceObject(pSurfObject, pResDesc);
}

cudaError_t Redirector::cudaDestroySurfaceObject(cudaSurfaceObject_t surfObject)
{
    return this->m_cudaDestroySurfaceObject(surfObject);
}

cudaError_t Redirector::cudaGetSurfaceObjectResourceDesc(struct cudaResourceDesc * pResDesc, cudaSurfaceObject_t surfObject)
{
    return this->m_cudaGetSurfaceObjectResourceDesc(pResDesc, surfObject);
}

cudaError_t Redirector::cudaDriverGetVersion(int * driverVersion)
{
    return this->m_cudaDriverGetVersion(driverVersion);
}

cudaError_t Redirector::cudaRuntimeGetVersion(int * runtimeVersion)
{
    return this->m_cudaRuntimeGetVersion(runtimeVersion);
}

cudaError_t Redirector::cudaGetExportTable(const void ** ppExportTable, const cudaUUID_t * pExportTableId)
{
    return this->m_cudaGetExportTable(ppExportTable, pExportTableId);
}

Redirector::Redirector()
{
    m_cudaDeviceReset = reinterpret_cast<APIcudaDeviceReset>(dlsym(RTLD_NEXT, "cudaDeviceReset"));
    m_cudaDeviceSynchronize = reinterpret_cast<APIcudaDeviceSynchronize>(dlsym(RTLD_NEXT, "cudaDeviceSynchronize"));
    m_cudaDeviceSetLimit = reinterpret_cast<APIcudaDeviceSetLimit>(dlsym(RTLD_NEXT, "cudaDeviceSetLimit"));
    m_cudaDeviceGetLimit = reinterpret_cast<APIcudaDeviceGetLimit>(dlsym(RTLD_NEXT, "cudaDeviceGetLimit"));
    m_cudaDeviceGetCacheConfig = reinterpret_cast<APIcudaDeviceGetCacheConfig>(dlsym(RTLD_NEXT, "cudaDeviceGetCacheConfig"));
    m_cudaDeviceSetCacheConfig = reinterpret_cast<APIcudaDeviceSetCacheConfig>(dlsym(RTLD_NEXT, "cudaDeviceSetCacheConfig"));
    m_cudaDeviceGetSharedMemConfig = reinterpret_cast<APIcudaDeviceGetSharedMemConfig>(dlsym(RTLD_NEXT, "cudaDeviceGetSharedMemConfig"));
    m_cudaDeviceSetSharedMemConfig = reinterpret_cast<APIcudaDeviceSetSharedMemConfig>(dlsym(RTLD_NEXT, "cudaDeviceSetSharedMemConfig"));
    m_cudaDeviceGetByPCIBusId = reinterpret_cast<APIcudaDeviceGetByPCIBusId>(dlsym(RTLD_NEXT, "cudaDeviceGetByPCIBusId"));
    m_cudaDeviceGetPCIBusId = reinterpret_cast<APIcudaDeviceGetPCIBusId>(dlsym(RTLD_NEXT, "cudaDeviceGetPCIBusId"));
    m_cudaIpcGetEventHandle = reinterpret_cast<APIcudaIpcGetEventHandle>(dlsym(RTLD_NEXT, "cudaIpcGetEventHandle"));
    m_cudaIpcOpenEventHandle = reinterpret_cast<APIcudaIpcOpenEventHandle>(dlsym(RTLD_NEXT, "cudaIpcOpenEventHandle"));
    m_cudaIpcGetMemHandle = reinterpret_cast<APIcudaIpcGetMemHandle>(dlsym(RTLD_NEXT, "cudaIpcGetMemHandle"));
    m_cudaIpcOpenMemHandle = reinterpret_cast<APIcudaIpcOpenMemHandle>(dlsym(RTLD_NEXT, "cudaIpcOpenMemHandle"));
    m_cudaIpcCloseMemHandle = reinterpret_cast<APIcudaIpcCloseMemHandle>(dlsym(RTLD_NEXT, "cudaIpcCloseMemHandle"));
    m_cudaThreadExit = reinterpret_cast<APIcudaThreadExit>(dlsym(RTLD_NEXT, "cudaThreadExit"));
    m_cudaThreadSynchronize = reinterpret_cast<APIcudaThreadSynchronize>(dlsym(RTLD_NEXT, "cudaThreadSynchronize"));
    m_cudaThreadSetLimit = reinterpret_cast<APIcudaThreadSetLimit>(dlsym(RTLD_NEXT, "cudaThreadSetLimit"));
    m_cudaThreadGetLimit = reinterpret_cast<APIcudaThreadGetLimit>(dlsym(RTLD_NEXT, "cudaThreadGetLimit"));
    m_cudaThreadGetCacheConfig = reinterpret_cast<APIcudaThreadGetCacheConfig>(dlsym(RTLD_NEXT, "cudaThreadGetCacheConfig"));
    m_cudaThreadSetCacheConfig = reinterpret_cast<APIcudaThreadSetCacheConfig>(dlsym(RTLD_NEXT, "cudaThreadSetCacheConfig"));
    m_cudaGetLastError = reinterpret_cast<APIcudaGetLastError>(dlsym(RTLD_NEXT, "cudaGetLastError"));
    m_cudaPeekAtLastError = reinterpret_cast<APIcudaPeekAtLastError>(dlsym(RTLD_NEXT, "cudaPeekAtLastError"));
    m_cudaGetErrorString = reinterpret_cast<APIcudaGetErrorString>(dlsym(RTLD_NEXT, "cudaGetErrorString"));
    m_cudaGetDeviceCount = reinterpret_cast<APIcudaGetDeviceCount>(dlsym(RTLD_NEXT, "cudaGetDeviceCount"));
    m_cudaGetDeviceProperties = reinterpret_cast<APIcudaGetDeviceProperties>(dlsym(RTLD_NEXT, "cudaGetDeviceProperties"));
    m_cudaDeviceGetAttribute = reinterpret_cast<APIcudaDeviceGetAttribute>(dlsym(RTLD_NEXT, "cudaDeviceGetAttribute"));
    m_cudaChooseDevice = reinterpret_cast<APIcudaChooseDevice>(dlsym(RTLD_NEXT, "cudaChooseDevice"));
    m_cudaSetDevice = reinterpret_cast<APIcudaSetDevice>(dlsym(RTLD_NEXT, "cudaSetDevice"));
    m_cudaGetDevice = reinterpret_cast<APIcudaGetDevice>(dlsym(RTLD_NEXT, "cudaGetDevice"));
    m_cudaSetValidDevices = reinterpret_cast<APIcudaSetValidDevices>(dlsym(RTLD_NEXT, "cudaSetValidDevices"));
    m_cudaSetDeviceFlags = reinterpret_cast<APIcudaSetDeviceFlags>(dlsym(RTLD_NEXT, "cudaSetDeviceFlags"));
    m_cudaStreamCreate = reinterpret_cast<APIcudaStreamCreate>(dlsym(RTLD_NEXT, "cudaStreamCreate"));
    m_cudaStreamCreateWithFlags = reinterpret_cast<APIcudaStreamCreateWithFlags>(dlsym(RTLD_NEXT, "cudaStreamCreateWithFlags"));
    m_cudaStreamDestroy = reinterpret_cast<APIcudaStreamDestroy>(dlsym(RTLD_NEXT, "cudaStreamDestroy"));
    m_cudaStreamWaitEvent = reinterpret_cast<APIcudaStreamWaitEvent>(dlsym(RTLD_NEXT, "cudaStreamWaitEvent"));
    m_cudaStreamAddCallback = reinterpret_cast<APIcudaStreamAddCallback>(dlsym(RTLD_NEXT, "cudaStreamAddCallback"));
    m_cudaStreamSynchronize = reinterpret_cast<APIcudaStreamSynchronize>(dlsym(RTLD_NEXT, "cudaStreamSynchronize"));
    m_cudaStreamQuery = reinterpret_cast<APIcudaStreamQuery>(dlsym(RTLD_NEXT, "cudaStreamQuery"));
    m_cudaEventCreate = reinterpret_cast<APIcudaEventCreate>(dlsym(RTLD_NEXT, "cudaEventCreate"));
    m_cudaEventCreateWithFlags = reinterpret_cast<APIcudaEventCreateWithFlags>(dlsym(RTLD_NEXT, "cudaEventCreateWithFlags"));
    m_cudaEventRecord = reinterpret_cast<APIcudaEventRecord>(dlsym(RTLD_NEXT, "cudaEventRecord"));
    m_cudaEventQuery = reinterpret_cast<APIcudaEventQuery>(dlsym(RTLD_NEXT, "cudaEventQuery"));
    m_cudaEventSynchronize = reinterpret_cast<APIcudaEventSynchronize>(dlsym(RTLD_NEXT, "cudaEventSynchronize"));
    m_cudaEventDestroy = reinterpret_cast<APIcudaEventDestroy>(dlsym(RTLD_NEXT, "cudaEventDestroy"));
    m_cudaEventElapsedTime = reinterpret_cast<APIcudaEventElapsedTime>(dlsym(RTLD_NEXT, "cudaEventElapsedTime"));
    m_cudaConfigureCall = reinterpret_cast<APIcudaConfigureCall>(dlsym(RTLD_NEXT, "cudaConfigureCall"));
    m_cudaSetupArgument = reinterpret_cast<APIcudaSetupArgument>(dlsym(RTLD_NEXT, "cudaSetupArgument"));
    m_cudaFuncSetCacheConfig = reinterpret_cast<APIcudaFuncSetCacheConfig>(dlsym(RTLD_NEXT, "cudaFuncSetCacheConfig"));
    m_cudaFuncSetSharedMemConfig = reinterpret_cast<APIcudaFuncSetSharedMemConfig>(dlsym(RTLD_NEXT, "cudaFuncSetSharedMemConfig"));
    m_cudaLaunch = reinterpret_cast<APIcudaLaunch>(dlsym(RTLD_NEXT, "cudaLaunch"));
    m_cudaFuncGetAttributes = reinterpret_cast<APIcudaFuncGetAttributes>(dlsym(RTLD_NEXT, "cudaFuncGetAttributes"));
    m_cudaSetDoubleForDevice = reinterpret_cast<APIcudaSetDoubleForDevice>(dlsym(RTLD_NEXT, "cudaSetDoubleForDevice"));
    m_cudaSetDoubleForHost = reinterpret_cast<APIcudaSetDoubleForHost>(dlsym(RTLD_NEXT, "cudaSetDoubleForHost"));
    m_cudaMalloc = reinterpret_cast<APIcudaMalloc>(dlsym(RTLD_NEXT, "cudaMalloc"));
    m_cudaMallocHost = reinterpret_cast<APIcudaMallocHost>(dlsym(RTLD_NEXT, "cudaMallocHost"));
    m_cudaMallocPitch = reinterpret_cast<APIcudaMallocPitch>(dlsym(RTLD_NEXT, "cudaMallocPitch"));
    m_cudaMallocArray = reinterpret_cast<APIcudaMallocArray>(dlsym(RTLD_NEXT, "cudaMallocArray"));
    m_cudaFree = reinterpret_cast<APIcudaFree>(dlsym(RTLD_NEXT, "cudaFree"));
    m_cudaFreeHost = reinterpret_cast<APIcudaFreeHost>(dlsym(RTLD_NEXT, "cudaFreeHost"));
    m_cudaFreeArray = reinterpret_cast<APIcudaFreeArray>(dlsym(RTLD_NEXT, "cudaFreeArray"));
    m_cudaFreeMipmappedArray = reinterpret_cast<APIcudaFreeMipmappedArray>(dlsym(RTLD_NEXT, "cudaFreeMipmappedArray"));
    m_cudaHostAlloc = reinterpret_cast<APIcudaHostAlloc>(dlsym(RTLD_NEXT, "cudaHostAlloc"));
    m_cudaHostRegister = reinterpret_cast<APIcudaHostRegister>(dlsym(RTLD_NEXT, "cudaHostRegister"));
    m_cudaHostUnregister = reinterpret_cast<APIcudaHostUnregister>(dlsym(RTLD_NEXT, "cudaHostUnregister"));
    m_cudaHostGetDevicePointer = reinterpret_cast<APIcudaHostGetDevicePointer>(dlsym(RTLD_NEXT, "cudaHostGetDevicePointer"));
    m_cudaHostGetFlags = reinterpret_cast<APIcudaHostGetFlags>(dlsym(RTLD_NEXT, "cudaHostGetFlags"));
    m_cudaMalloc3D = reinterpret_cast<APIcudaMalloc3D>(dlsym(RTLD_NEXT, "cudaMalloc3D"));
    m_cudaMalloc3DArray = reinterpret_cast<APIcudaMalloc3DArray>(dlsym(RTLD_NEXT, "cudaMalloc3DArray"));
    m_cudaMallocMipmappedArray = reinterpret_cast<APIcudaMallocMipmappedArray>(dlsym(RTLD_NEXT, "cudaMallocMipmappedArray"));
    m_cudaGetMipmappedArrayLevel = reinterpret_cast<APIcudaGetMipmappedArrayLevel>(dlsym(RTLD_NEXT, "cudaGetMipmappedArrayLevel"));
    m_cudaMemcpy3D = reinterpret_cast<APIcudaMemcpy3D>(dlsym(RTLD_NEXT, "cudaMemcpy3D"));
    m_cudaMemcpy3DPeer = reinterpret_cast<APIcudaMemcpy3DPeer>(dlsym(RTLD_NEXT, "cudaMemcpy3DPeer"));
    m_cudaMemcpy3DAsync = reinterpret_cast<APIcudaMemcpy3DAsync>(dlsym(RTLD_NEXT, "cudaMemcpy3DAsync"));
    m_cudaMemcpy3DPeerAsync = reinterpret_cast<APIcudaMemcpy3DPeerAsync>(dlsym(RTLD_NEXT, "cudaMemcpy3DPeerAsync"));
    m_cudaMemGetInfo = reinterpret_cast<APIcudaMemGetInfo>(dlsym(RTLD_NEXT, "cudaMemGetInfo"));
    m_cudaArrayGetInfo = reinterpret_cast<APIcudaArrayGetInfo>(dlsym(RTLD_NEXT, "cudaArrayGetInfo"));
    m_cudaMemcpy = reinterpret_cast<APIcudaMemcpy>(dlsym(RTLD_NEXT, "cudaMemcpy"));
    m_cudaMemcpyPeer = reinterpret_cast<APIcudaMemcpyPeer>(dlsym(RTLD_NEXT, "cudaMemcpyPeer"));
    m_cudaMemcpyToArray = reinterpret_cast<APIcudaMemcpyToArray>(dlsym(RTLD_NEXT, "cudaMemcpyToArray"));
    m_cudaMemcpyFromArray = reinterpret_cast<APIcudaMemcpyFromArray>(dlsym(RTLD_NEXT, "cudaMemcpyFromArray"));
    m_cudaMemcpyArrayToArray = reinterpret_cast<APIcudaMemcpyArrayToArray>(dlsym(RTLD_NEXT, "cudaMemcpyArrayToArray"));
    m_cudaMemcpy2D = reinterpret_cast<APIcudaMemcpy2D>(dlsym(RTLD_NEXT, "cudaMemcpy2D"));
    m_cudaMemcpy2DToArray = reinterpret_cast<APIcudaMemcpy2DToArray>(dlsym(RTLD_NEXT, "cudaMemcpy2DToArray"));
    m_cudaMemcpy2DFromArray = reinterpret_cast<APIcudaMemcpy2DFromArray>(dlsym(RTLD_NEXT, "cudaMemcpy2DFromArray"));
    m_cudaMemcpy2DArrayToArray = reinterpret_cast<APIcudaMemcpy2DArrayToArray>(dlsym(RTLD_NEXT, "cudaMemcpy2DArrayToArray"));
    m_cudaMemcpyToSymbol = reinterpret_cast<APIcudaMemcpyToSymbol>(dlsym(RTLD_NEXT, "cudaMemcpyToSymbol"));
    m_cudaMemcpyFromSymbol = reinterpret_cast<APIcudaMemcpyFromSymbol>(dlsym(RTLD_NEXT, "cudaMemcpyFromSymbol"));
    m_cudaMemcpyAsync = reinterpret_cast<APIcudaMemcpyAsync>(dlsym(RTLD_NEXT, "cudaMemcpyAsync"));
    m_cudaMemcpyPeerAsync = reinterpret_cast<APIcudaMemcpyPeerAsync>(dlsym(RTLD_NEXT, "cudaMemcpyPeerAsync"));
    m_cudaMemcpyToArrayAsync = reinterpret_cast<APIcudaMemcpyToArrayAsync>(dlsym(RTLD_NEXT, "cudaMemcpyToArrayAsync"));
    m_cudaMemcpyFromArrayAsync = reinterpret_cast<APIcudaMemcpyFromArrayAsync>(dlsym(RTLD_NEXT, "cudaMemcpyFromArrayAsync"));
    m_cudaMemcpy2DAsync = reinterpret_cast<APIcudaMemcpy2DAsync>(dlsym(RTLD_NEXT, "cudaMemcpy2DAsync"));
    m_cudaMemcpy2DToArrayAsync = reinterpret_cast<APIcudaMemcpy2DToArrayAsync>(dlsym(RTLD_NEXT, "cudaMemcpy2DToArrayAsync"));
    m_cudaMemcpy2DFromArrayAsync = reinterpret_cast<APIcudaMemcpy2DFromArrayAsync>(dlsym(RTLD_NEXT, "cudaMemcpy2DFromArrayAsync"));
    m_cudaMemcpyToSymbolAsync = reinterpret_cast<APIcudaMemcpyToSymbolAsync>(dlsym(RTLD_NEXT, "cudaMemcpyToSymbolAsync"));
    m_cudaMemcpyFromSymbolAsync = reinterpret_cast<APIcudaMemcpyFromSymbolAsync>(dlsym(RTLD_NEXT, "cudaMemcpyFromSymbolAsync"));
    m_cudaMemset = reinterpret_cast<APIcudaMemset>(dlsym(RTLD_NEXT, "cudaMemset"));
    m_cudaMemset2D = reinterpret_cast<APIcudaMemset2D>(dlsym(RTLD_NEXT, "cudaMemset2D"));
    m_cudaMemset3D = reinterpret_cast<APIcudaMemset3D>(dlsym(RTLD_NEXT, "cudaMemset3D"));
    m_cudaMemsetAsync = reinterpret_cast<APIcudaMemsetAsync>(dlsym(RTLD_NEXT, "cudaMemsetAsync"));
    m_cudaMemset2DAsync = reinterpret_cast<APIcudaMemset2DAsync>(dlsym(RTLD_NEXT, "cudaMemset2DAsync"));
    m_cudaMemset3DAsync = reinterpret_cast<APIcudaMemset3DAsync>(dlsym(RTLD_NEXT, "cudaMemset3DAsync"));
    m_cudaGetSymbolAddress = reinterpret_cast<APIcudaGetSymbolAddress>(dlsym(RTLD_NEXT, "cudaGetSymbolAddress"));
    m_cudaGetSymbolSize = reinterpret_cast<APIcudaGetSymbolSize>(dlsym(RTLD_NEXT, "cudaGetSymbolSize"));
    m_cudaPointerGetAttributes = reinterpret_cast<APIcudaPointerGetAttributes>(dlsym(RTLD_NEXT, "cudaPointerGetAttributes"));
    m_cudaDeviceCanAccessPeer = reinterpret_cast<APIcudaDeviceCanAccessPeer>(dlsym(RTLD_NEXT, "cudaDeviceCanAccessPeer"));
    m_cudaDeviceEnablePeerAccess = reinterpret_cast<APIcudaDeviceEnablePeerAccess>(dlsym(RTLD_NEXT, "cudaDeviceEnablePeerAccess"));
    m_cudaDeviceDisablePeerAccess = reinterpret_cast<APIcudaDeviceDisablePeerAccess>(dlsym(RTLD_NEXT, "cudaDeviceDisablePeerAccess"));
    m_cudaGraphicsUnregisterResource = reinterpret_cast<APIcudaGraphicsUnregisterResource>(dlsym(RTLD_NEXT, "cudaGraphicsUnregisterResource"));
    m_cudaGraphicsResourceSetMapFlags = reinterpret_cast<APIcudaGraphicsResourceSetMapFlags>(dlsym(RTLD_NEXT, "cudaGraphicsResourceSetMapFlags"));
    m_cudaGraphicsMapResources = reinterpret_cast<APIcudaGraphicsMapResources>(dlsym(RTLD_NEXT, "cudaGraphicsMapResources"));
    m_cudaGraphicsUnmapResources = reinterpret_cast<APIcudaGraphicsUnmapResources>(dlsym(RTLD_NEXT, "cudaGraphicsUnmapResources"));
    m_cudaGraphicsResourceGetMappedPointer = reinterpret_cast<APIcudaGraphicsResourceGetMappedPointer>(dlsym(RTLD_NEXT, "cudaGraphicsResourceGetMappedPointer"));
    m_cudaGraphicsSubResourceGetMappedArray = reinterpret_cast<APIcudaGraphicsSubResourceGetMappedArray>(dlsym(RTLD_NEXT, "cudaGraphicsSubResourceGetMappedArray"));
    m_cudaGraphicsResourceGetMappedMipmappedArray = reinterpret_cast<APIcudaGraphicsResourceGetMappedMipmappedArray>(dlsym(RTLD_NEXT, "cudaGraphicsResourceGetMappedMipmappedArray"));
    m_cudaGetChannelDesc = reinterpret_cast<APIcudaGetChannelDesc>(dlsym(RTLD_NEXT, "cudaGetChannelDesc"));
    m_cudaCreateChannelDesc = reinterpret_cast<APIcudaCreateChannelDesc>(dlsym(RTLD_NEXT, "cudaCreateChannelDesc"));
    m_cudaBindTexture = reinterpret_cast<APIcudaBindTexture>(dlsym(RTLD_NEXT, "cudaBindTexture"));
    m_cudaBindTexture2D = reinterpret_cast<APIcudaBindTexture2D>(dlsym(RTLD_NEXT, "cudaBindTexture2D"));
    m_cudaBindTextureToArray = reinterpret_cast<APIcudaBindTextureToArray>(dlsym(RTLD_NEXT, "cudaBindTextureToArray"));
    m_cudaBindTextureToMipmappedArray = reinterpret_cast<APIcudaBindTextureToMipmappedArray>(dlsym(RTLD_NEXT, "cudaBindTextureToMipmappedArray"));
    m_cudaUnbindTexture = reinterpret_cast<APIcudaUnbindTexture>(dlsym(RTLD_NEXT, "cudaUnbindTexture"));
    m_cudaGetTextureAlignmentOffset = reinterpret_cast<APIcudaGetTextureAlignmentOffset>(dlsym(RTLD_NEXT, "cudaGetTextureAlignmentOffset"));
    m_cudaGetTextureReference = reinterpret_cast<APIcudaGetTextureReference>(dlsym(RTLD_NEXT, "cudaGetTextureReference"));
    m_cudaBindSurfaceToArray = reinterpret_cast<APIcudaBindSurfaceToArray>(dlsym(RTLD_NEXT, "cudaBindSurfaceToArray"));
    m_cudaGetSurfaceReference = reinterpret_cast<APIcudaGetSurfaceReference>(dlsym(RTLD_NEXT, "cudaGetSurfaceReference"));
    m_cudaCreateTextureObject = reinterpret_cast<APIcudaCreateTextureObject>(dlsym(RTLD_NEXT, "cudaCreateTextureObject"));
    m_cudaDestroyTextureObject = reinterpret_cast<APIcudaDestroyTextureObject>(dlsym(RTLD_NEXT, "cudaDestroyTextureObject"));
    m_cudaGetTextureObjectResourceDesc = reinterpret_cast<APIcudaGetTextureObjectResourceDesc>(dlsym(RTLD_NEXT, "cudaGetTextureObjectResourceDesc"));
    m_cudaGetTextureObjectTextureDesc = reinterpret_cast<APIcudaGetTextureObjectTextureDesc>(dlsym(RTLD_NEXT, "cudaGetTextureObjectTextureDesc"));
    m_cudaGetTextureObjectResourceViewDesc = reinterpret_cast<APIcudaGetTextureObjectResourceViewDesc>(dlsym(RTLD_NEXT, "cudaGetTextureObjectResourceViewDesc"));
    m_cudaCreateSurfaceObject = reinterpret_cast<APIcudaCreateSurfaceObject>(dlsym(RTLD_NEXT, "cudaCreateSurfaceObject"));
    m_cudaDestroySurfaceObject = reinterpret_cast<APIcudaDestroySurfaceObject>(dlsym(RTLD_NEXT, "cudaDestroySurfaceObject"));
    m_cudaGetSurfaceObjectResourceDesc = reinterpret_cast<APIcudaGetSurfaceObjectResourceDesc>(dlsym(RTLD_NEXT, "cudaGetSurfaceObjectResourceDesc"));
    m_cudaDriverGetVersion = reinterpret_cast<APIcudaDriverGetVersion>(dlsym(RTLD_NEXT, "cudaDriverGetVersion"));
    m_cudaRuntimeGetVersion = reinterpret_cast<APIcudaRuntimeGetVersion>(dlsym(RTLD_NEXT, "cudaRuntimeGetVersion"));
    m_cudaGetExportTable = reinterpret_cast<APIcudaGetExportTable>(dlsym(RTLD_NEXT, "cudaGetExportTable"));
}
} }  // namespace gnode::hooks

extern "C" {

cudaError_t cudaDeviceReset()
{
    return gnode::hooks::MainLoop::instance().cudaDeviceReset();
}

cudaError_t cudaDeviceSynchronize()
{
    return gnode::hooks::MainLoop::instance().cudaDeviceSynchronize();
}

cudaError_t cudaDeviceSetLimit(enum cudaLimit limit, size_t value)
{
    return gnode::hooks::MainLoop::instance().cudaDeviceSetLimit(limit, value);
}

cudaError_t cudaDeviceGetLimit(size_t * pValue, enum cudaLimit limit)
{
    return gnode::hooks::MainLoop::instance().cudaDeviceGetLimit(pValue, limit);
}

cudaError_t cudaDeviceGetCacheConfig(enum cudaFuncCache * pCacheConfig)
{
    return gnode::hooks::MainLoop::instance().cudaDeviceGetCacheConfig(pCacheConfig);
}

cudaError_t cudaDeviceSetCacheConfig(enum cudaFuncCache cacheConfig)
{
    return gnode::hooks::MainLoop::instance().cudaDeviceSetCacheConfig(cacheConfig);
}

cudaError_t cudaDeviceGetSharedMemConfig(enum cudaSharedMemConfig * pConfig)
{
    return gnode::hooks::MainLoop::instance().cudaDeviceGetSharedMemConfig(pConfig);
}

cudaError_t cudaDeviceSetSharedMemConfig(enum cudaSharedMemConfig config)
{
    return gnode::hooks::MainLoop::instance().cudaDeviceSetSharedMemConfig(config);
}

cudaError_t cudaDeviceGetByPCIBusId(int * device, char * pciBusId)
{
    return gnode::hooks::MainLoop::instance().cudaDeviceGetByPCIBusId(device, pciBusId);
}

cudaError_t cudaDeviceGetPCIBusId(char * pciBusId, int len, int device)
{
    return gnode::hooks::MainLoop::instance().cudaDeviceGetPCIBusId(pciBusId, len, device);
}

cudaError_t cudaIpcGetEventHandle(cudaIpcEventHandle_t * handle, cudaEvent_t event)
{
    return gnode::hooks::MainLoop::instance().cudaIpcGetEventHandle(handle, event);
}

cudaError_t cudaIpcOpenEventHandle(cudaEvent_t * event, cudaIpcEventHandle_t handle)
{
    return gnode::hooks::MainLoop::instance().cudaIpcOpenEventHandle(event, handle);
}

cudaError_t cudaIpcGetMemHandle(cudaIpcMemHandle_t * handle, void * devPtr)
{
    return gnode::hooks::MainLoop::instance().cudaIpcGetMemHandle(handle, devPtr);
}

cudaError_t cudaIpcOpenMemHandle(void ** devPtr, cudaIpcMemHandle_t handle, unsigned int flags)
{
    return gnode::hooks::MainLoop::instance().cudaIpcOpenMemHandle(devPtr, handle, flags);
}

cudaError_t cudaIpcCloseMemHandle(void * devPtr)
{
    return gnode::hooks::MainLoop::instance().cudaIpcCloseMemHandle(devPtr);
}

cudaError_t cudaThreadExit()
{
    return gnode::hooks::MainLoop::instance().cudaThreadExit();
}

cudaError_t cudaThreadSynchronize()
{
    return gnode::hooks::MainLoop::instance().cudaThreadSynchronize();
}

cudaError_t cudaThreadSetLimit(enum cudaLimit limit, size_t value)
{
    return gnode::hooks::MainLoop::instance().cudaThreadSetLimit(limit, value);
}

cudaError_t cudaThreadGetLimit(size_t * pValue, enum cudaLimit limit)
{
    return gnode::hooks::MainLoop::instance().cudaThreadGetLimit(pValue, limit);
}

cudaError_t cudaThreadGetCacheConfig(enum cudaFuncCache * pCacheConfig)
{
    return gnode::hooks::MainLoop::instance().cudaThreadGetCacheConfig(pCacheConfig);
}

cudaError_t cudaThreadSetCacheConfig(enum cudaFuncCache cacheConfig)
{
    return gnode::hooks::MainLoop::instance().cudaThreadSetCacheConfig(cacheConfig);
}

cudaError_t cudaGetLastError()
{
    return gnode::hooks::MainLoop::instance().cudaGetLastError();
}

cudaError_t cudaPeekAtLastError()
{
    return gnode::hooks::MainLoop::instance().cudaPeekAtLastError();
}

const char * cudaGetErrorString(cudaError_t error)
{
    return gnode::hooks::MainLoop::instance().cudaGetErrorString(error);
}

cudaError_t cudaGetDeviceCount(int * count)
{
    return gnode::hooks::MainLoop::instance().cudaGetDeviceCount(count);
}

cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp * prop, int device)
{
    return gnode::hooks::MainLoop::instance().cudaGetDeviceProperties(prop, device);
}

cudaError_t cudaDeviceGetAttribute(int * value, enum cudaDeviceAttr attr, int device)
{
    return gnode::hooks::MainLoop::instance().cudaDeviceGetAttribute(value, attr, device);
}

cudaError_t cudaChooseDevice(int * device, const struct cudaDeviceProp * prop)
{
    return gnode::hooks::MainLoop::instance().cudaChooseDevice(device, prop);
}

cudaError_t cudaSetDevice(int device)
{
    return gnode::hooks::MainLoop::instance().cudaSetDevice(device);
}

cudaError_t cudaGetDevice(int * device)
{
    return gnode::hooks::MainLoop::instance().cudaGetDevice(device);
}

cudaError_t cudaSetValidDevices(int * device_arr, int len)
{
    return gnode::hooks::MainLoop::instance().cudaSetValidDevices(device_arr, len);
}

cudaError_t cudaSetDeviceFlags(unsigned int flags)
{
    return gnode::hooks::MainLoop::instance().cudaSetDeviceFlags(flags);
}

cudaError_t cudaStreamCreate(cudaStream_t * pStream)
{
    return gnode::hooks::MainLoop::instance().cudaStreamCreate(pStream);
}

cudaError_t cudaStreamCreateWithFlags(cudaStream_t * pStream, unsigned int flags)
{
    return gnode::hooks::MainLoop::instance().cudaStreamCreateWithFlags(pStream, flags);
}

cudaError_t cudaStreamDestroy(cudaStream_t stream)
{
    return gnode::hooks::MainLoop::instance().cudaStreamDestroy(stream);
}

cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags)
{
    return gnode::hooks::MainLoop::instance().cudaStreamWaitEvent(stream, event, flags);
}

cudaError_t cudaStreamAddCallback(cudaStream_t stream, cudaStreamCallback_t callback, void * userData, unsigned int flags)
{
    return gnode::hooks::MainLoop::instance().cudaStreamAddCallback(stream, callback, userData, flags);
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream)
{
    return gnode::hooks::MainLoop::instance().cudaStreamSynchronize(stream);
}

cudaError_t cudaStreamQuery(cudaStream_t stream)
{
    return gnode::hooks::MainLoop::instance().cudaStreamQuery(stream);
}

cudaError_t cudaEventCreate(cudaEvent_t * event)
{
    return gnode::hooks::MainLoop::instance().cudaEventCreate(event);
}

cudaError_t cudaEventCreateWithFlags(cudaEvent_t * event, unsigned int flags)
{
    return gnode::hooks::MainLoop::instance().cudaEventCreateWithFlags(event, flags);
}

cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream)
{
    return gnode::hooks::MainLoop::instance().cudaEventRecord(event, stream);
}

cudaError_t cudaEventQuery(cudaEvent_t event)
{
    return gnode::hooks::MainLoop::instance().cudaEventQuery(event);
}

cudaError_t cudaEventSynchronize(cudaEvent_t event)
{
    return gnode::hooks::MainLoop::instance().cudaEventSynchronize(event);
}

cudaError_t cudaEventDestroy(cudaEvent_t event)
{
    return gnode::hooks::MainLoop::instance().cudaEventDestroy(event);
}

cudaError_t cudaEventElapsedTime(float * ms, cudaEvent_t start, cudaEvent_t end)
{
    return gnode::hooks::MainLoop::instance().cudaEventElapsedTime(ms, start, end);
}

cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream)
{
    return gnode::hooks::MainLoop::instance().cudaConfigureCall(gridDim, blockDim, sharedMem, stream);
}

cudaError_t cudaSetupArgument(const void * arg, size_t size, size_t offset)
{
    return gnode::hooks::MainLoop::instance().cudaSetupArgument(arg, size, offset);
}

cudaError_t cudaFuncSetCacheConfig(const void * func, enum cudaFuncCache cacheConfig)
{
    return gnode::hooks::MainLoop::instance().cudaFuncSetCacheConfig(func, cacheConfig);
}

cudaError_t cudaFuncSetSharedMemConfig(const void * func, enum cudaSharedMemConfig config)
{
    return gnode::hooks::MainLoop::instance().cudaFuncSetSharedMemConfig(func, config);
}

cudaError_t cudaLaunch(const void * func)
{
    return gnode::hooks::MainLoop::instance().cudaLaunch(func);
}

cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes * attr, const void * func)
{
    return gnode::hooks::MainLoop::instance().cudaFuncGetAttributes(attr, func);
}

cudaError_t cudaSetDoubleForDevice(double * d)
{
    return gnode::hooks::MainLoop::instance().cudaSetDoubleForDevice(d);
}

cudaError_t cudaSetDoubleForHost(double * d)
{
    return gnode::hooks::MainLoop::instance().cudaSetDoubleForHost(d);
}

cudaError_t cudaMalloc(void ** devPtr, size_t size)
{
    return gnode::hooks::MainLoop::instance().cudaMalloc(devPtr, size);
}

cudaError_t cudaMallocHost(void ** ptr, size_t size)
{
    return gnode::hooks::MainLoop::instance().cudaMallocHost(ptr, size);
}

cudaError_t cudaMallocPitch(void ** devPtr, size_t * pitch, size_t width, size_t height)
{
    return gnode::hooks::MainLoop::instance().cudaMallocPitch(devPtr, pitch, width, height);
}

cudaError_t cudaMallocArray(cudaArray_t * array, const struct cudaChannelFormatDesc * desc, size_t width, size_t height, unsigned int flags)
{
    return gnode::hooks::MainLoop::instance().cudaMallocArray(array, desc, width, height, flags);
}

cudaError_t cudaFree(void * devPtr)
{
    return gnode::hooks::MainLoop::instance().cudaFree(devPtr);
}

cudaError_t cudaFreeHost(void * ptr)
{
    return gnode::hooks::MainLoop::instance().cudaFreeHost(ptr);
}

cudaError_t cudaFreeArray(cudaArray_t array)
{
    return gnode::hooks::MainLoop::instance().cudaFreeArray(array);
}

cudaError_t cudaFreeMipmappedArray(cudaMipmappedArray_t mipmappedArray)
{
    return gnode::hooks::MainLoop::instance().cudaFreeMipmappedArray(mipmappedArray);
}

cudaError_t cudaHostAlloc(void ** pHost, size_t size, unsigned int flags)
{
    return gnode::hooks::MainLoop::instance().cudaHostAlloc(pHost, size, flags);
}

cudaError_t cudaHostRegister(void * ptr, size_t size, unsigned int flags)
{
    return gnode::hooks::MainLoop::instance().cudaHostRegister(ptr, size, flags);
}

cudaError_t cudaHostUnregister(void * ptr)
{
    return gnode::hooks::MainLoop::instance().cudaHostUnregister(ptr);
}

cudaError_t cudaHostGetDevicePointer(void ** pDevice, void * pHost, unsigned int flags)
{
    return gnode::hooks::MainLoop::instance().cudaHostGetDevicePointer(pDevice, pHost, flags);
}

cudaError_t cudaHostGetFlags(unsigned int * pFlags, void * pHost)
{
    return gnode::hooks::MainLoop::instance().cudaHostGetFlags(pFlags, pHost);
}

cudaError_t cudaMalloc3D(struct cudaPitchedPtr * pitchedDevPtr, struct cudaExtent extent)
{
    return gnode::hooks::MainLoop::instance().cudaMalloc3D(pitchedDevPtr, extent);
}

cudaError_t cudaMalloc3DArray(cudaArray_t * array, const struct cudaChannelFormatDesc * desc, struct cudaExtent extent, unsigned int flags)
{
    return gnode::hooks::MainLoop::instance().cudaMalloc3DArray(array, desc, extent, flags);
}

cudaError_t cudaMallocMipmappedArray(cudaMipmappedArray_t * mipmappedArray, const struct cudaChannelFormatDesc * desc, struct cudaExtent extent, unsigned int numLevels, unsigned int flags)
{
    return gnode::hooks::MainLoop::instance().cudaMallocMipmappedArray(mipmappedArray, desc, extent, numLevels, flags);
}

cudaError_t cudaGetMipmappedArrayLevel(cudaArray_t * levelArray, cudaMipmappedArray_const_t mipmappedArray, unsigned int level)
{
    return gnode::hooks::MainLoop::instance().cudaGetMipmappedArrayLevel(levelArray, mipmappedArray, level);
}

cudaError_t cudaMemcpy3D(const struct cudaMemcpy3DParms * p)
{
    return gnode::hooks::MainLoop::instance().cudaMemcpy3D(p);
}

cudaError_t cudaMemcpy3DPeer(const struct cudaMemcpy3DPeerParms * p)
{
    return gnode::hooks::MainLoop::instance().cudaMemcpy3DPeer(p);
}

cudaError_t cudaMemcpy3DAsync(const struct cudaMemcpy3DParms * p, cudaStream_t stream)
{
    return gnode::hooks::MainLoop::instance().cudaMemcpy3DAsync(p, stream);
}

cudaError_t cudaMemcpy3DPeerAsync(const struct cudaMemcpy3DPeerParms * p, cudaStream_t stream)
{
    return gnode::hooks::MainLoop::instance().cudaMemcpy3DPeerAsync(p, stream);
}

cudaError_t cudaMemGetInfo(size_t * free, size_t * total)
{
    return gnode::hooks::MainLoop::instance().cudaMemGetInfo(free, total);
}

cudaError_t cudaArrayGetInfo(struct cudaChannelFormatDesc * desc, struct cudaExtent * extent, unsigned int * flags, cudaArray_t array)
{
    return gnode::hooks::MainLoop::instance().cudaArrayGetInfo(desc, extent, flags, array);
}

cudaError_t cudaMemcpy(void * dst, const void * src, size_t count, enum cudaMemcpyKind kind)
{
    return gnode::hooks::MainLoop::instance().cudaMemcpy(dst, src, count, kind);
}

cudaError_t cudaMemcpyPeer(void * dst, int dstDevice, const void * src, int srcDevice, size_t count)
{
    return gnode::hooks::MainLoop::instance().cudaMemcpyPeer(dst, dstDevice, src, srcDevice, count);
}

cudaError_t cudaMemcpyToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t count, enum cudaMemcpyKind kind)
{
    return gnode::hooks::MainLoop::instance().cudaMemcpyToArray(dst, wOffset, hOffset, src, count, kind);
}

cudaError_t cudaMemcpyFromArray(void * dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind)
{
    return gnode::hooks::MainLoop::instance().cudaMemcpyFromArray(dst, src, wOffset, hOffset, count, kind);
}

cudaError_t cudaMemcpyArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, enum cudaMemcpyKind kind)
{
    return gnode::hooks::MainLoop::instance().cudaMemcpyArrayToArray(dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc, count, kind);
}

cudaError_t cudaMemcpy2D(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind)
{
    return gnode::hooks::MainLoop::instance().cudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind);
}

cudaError_t cudaMemcpy2DToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind)
{
    return gnode::hooks::MainLoop::instance().cudaMemcpy2DToArray(dst, wOffset, hOffset, src, spitch, width, height, kind);
}

cudaError_t cudaMemcpy2DFromArray(void * dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind)
{
    return gnode::hooks::MainLoop::instance().cudaMemcpy2DFromArray(dst, dpitch, src, wOffset, hOffset, width, height, kind);
}

cudaError_t cudaMemcpy2DArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, enum cudaMemcpyKind kind)
{
    return gnode::hooks::MainLoop::instance().cudaMemcpy2DArrayToArray(dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc, width, height, kind);
}

cudaError_t cudaMemcpyToSymbol(const void * symbol, const void * src, size_t count, size_t offset, enum cudaMemcpyKind kind)
{
    return gnode::hooks::MainLoop::instance().cudaMemcpyToSymbol(symbol, src, count, offset, kind);
}

cudaError_t cudaMemcpyFromSymbol(void * dst, const void * symbol, size_t count, size_t offset, enum cudaMemcpyKind kind)
{
    return gnode::hooks::MainLoop::instance().cudaMemcpyFromSymbol(dst, symbol, count, offset, kind);
}

cudaError_t cudaMemcpyAsync(void * dst, const void * src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream)
{
    return gnode::hooks::MainLoop::instance().cudaMemcpyAsync(dst, src, count, kind, stream);
}

cudaError_t cudaMemcpyPeerAsync(void * dst, int dstDevice, const void * src, int srcDevice, size_t count, cudaStream_t stream)
{
    return gnode::hooks::MainLoop::instance().cudaMemcpyPeerAsync(dst, dstDevice, src, srcDevice, count, stream);
}

cudaError_t cudaMemcpyToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream)
{
    return gnode::hooks::MainLoop::instance().cudaMemcpyToArrayAsync(dst, wOffset, hOffset, src, count, kind, stream);
}

cudaError_t cudaMemcpyFromArrayAsync(void * dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream)
{
    return gnode::hooks::MainLoop::instance().cudaMemcpyFromArrayAsync(dst, src, wOffset, hOffset, count, kind, stream);
}

cudaError_t cudaMemcpy2DAsync(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream)
{
    return gnode::hooks::MainLoop::instance().cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, stream);
}

cudaError_t cudaMemcpy2DToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream)
{
    return gnode::hooks::MainLoop::instance().cudaMemcpy2DToArrayAsync(dst, wOffset, hOffset, src, spitch, width, height, kind, stream);
}

cudaError_t cudaMemcpy2DFromArrayAsync(void * dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream)
{
    return gnode::hooks::MainLoop::instance().cudaMemcpy2DFromArrayAsync(dst, dpitch, src, wOffset, hOffset, width, height, kind, stream);
}

cudaError_t cudaMemcpyToSymbolAsync(const void * symbol, const void * src, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream)
{
    return gnode::hooks::MainLoop::instance().cudaMemcpyToSymbolAsync(symbol, src, count, offset, kind, stream);
}

cudaError_t cudaMemcpyFromSymbolAsync(void * dst, const void * symbol, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream)
{
    return gnode::hooks::MainLoop::instance().cudaMemcpyFromSymbolAsync(dst, symbol, count, offset, kind, stream);
}

cudaError_t cudaMemset(void * devPtr, int value, size_t count)
{
    return gnode::hooks::MainLoop::instance().cudaMemset(devPtr, value, count);
}

cudaError_t cudaMemset2D(void * devPtr, size_t pitch, int value, size_t width, size_t height)
{
    return gnode::hooks::MainLoop::instance().cudaMemset2D(devPtr, pitch, value, width, height);
}

cudaError_t cudaMemset3D(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent)
{
    return gnode::hooks::MainLoop::instance().cudaMemset3D(pitchedDevPtr, value, extent);
}

cudaError_t cudaMemsetAsync(void * devPtr, int value, size_t count, cudaStream_t stream)
{
    return gnode::hooks::MainLoop::instance().cudaMemsetAsync(devPtr, value, count, stream);
}

cudaError_t cudaMemset2DAsync(void * devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream)
{
    return gnode::hooks::MainLoop::instance().cudaMemset2DAsync(devPtr, pitch, value, width, height, stream);
}

cudaError_t cudaMemset3DAsync(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent, cudaStream_t stream)
{
    return gnode::hooks::MainLoop::instance().cudaMemset3DAsync(pitchedDevPtr, value, extent, stream);
}

cudaError_t cudaGetSymbolAddress(void ** devPtr, const void * symbol)
{
    return gnode::hooks::MainLoop::instance().cudaGetSymbolAddress(devPtr, symbol);
}

cudaError_t cudaGetSymbolSize(size_t * size, const void * symbol)
{
    return gnode::hooks::MainLoop::instance().cudaGetSymbolSize(size, symbol);
}

cudaError_t cudaPointerGetAttributes(struct cudaPointerAttributes * attributes, const void * ptr)
{
    return gnode::hooks::MainLoop::instance().cudaPointerGetAttributes(attributes, ptr);
}

cudaError_t cudaDeviceCanAccessPeer(int * canAccessPeer, int device, int peerDevice)
{
    return gnode::hooks::MainLoop::instance().cudaDeviceCanAccessPeer(canAccessPeer, device, peerDevice);
}

cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags)
{
    return gnode::hooks::MainLoop::instance().cudaDeviceEnablePeerAccess(peerDevice, flags);
}

cudaError_t cudaDeviceDisablePeerAccess(int peerDevice)
{
    return gnode::hooks::MainLoop::instance().cudaDeviceDisablePeerAccess(peerDevice);
}

cudaError_t cudaGraphicsUnregisterResource(cudaGraphicsResource_t resource)
{
    return gnode::hooks::MainLoop::instance().cudaGraphicsUnregisterResource(resource);
}

cudaError_t cudaGraphicsResourceSetMapFlags(cudaGraphicsResource_t resource, unsigned int flags)
{
    return gnode::hooks::MainLoop::instance().cudaGraphicsResourceSetMapFlags(resource, flags);
}

cudaError_t cudaGraphicsMapResources(int count, cudaGraphicsResource_t * resources, cudaStream_t stream)
{
    return gnode::hooks::MainLoop::instance().cudaGraphicsMapResources(count, resources, stream);
}

cudaError_t cudaGraphicsUnmapResources(int count, cudaGraphicsResource_t * resources, cudaStream_t stream)
{
    return gnode::hooks::MainLoop::instance().cudaGraphicsUnmapResources(count, resources, stream);
}

cudaError_t cudaGraphicsResourceGetMappedPointer(void ** devPtr, size_t * size, cudaGraphicsResource_t resource)
{
    return gnode::hooks::MainLoop::instance().cudaGraphicsResourceGetMappedPointer(devPtr, size, resource);
}

cudaError_t cudaGraphicsSubResourceGetMappedArray(cudaArray_t * array, cudaGraphicsResource_t resource, unsigned int arrayIndex, unsigned int mipLevel)
{
    return gnode::hooks::MainLoop::instance().cudaGraphicsSubResourceGetMappedArray(array, resource, arrayIndex, mipLevel);
}

cudaError_t cudaGraphicsResourceGetMappedMipmappedArray(cudaMipmappedArray_t * mipmappedArray, cudaGraphicsResource_t resource)
{
    return gnode::hooks::MainLoop::instance().cudaGraphicsResourceGetMappedMipmappedArray(mipmappedArray, resource);
}

cudaError_t cudaGetChannelDesc(struct cudaChannelFormatDesc * desc, cudaArray_const_t array)
{
    return gnode::hooks::MainLoop::instance().cudaGetChannelDesc(desc, array);
}

struct cudaChannelFormatDesc cudaCreateChannelDesc(int x, int y, int z, int w, enum cudaChannelFormatKind f)
{
    return gnode::hooks::MainLoop::instance().cudaCreateChannelDesc(x, y, z, w, f);
}

cudaError_t cudaBindTexture(size_t * offset, const struct textureReference * texref, const void * devPtr, const struct cudaChannelFormatDesc * desc, size_t size)
{
    return gnode::hooks::MainLoop::instance().cudaBindTexture(offset, texref, devPtr, desc, size);
}

cudaError_t cudaBindTexture2D(size_t * offset, const struct textureReference * texref, const void * devPtr, const struct cudaChannelFormatDesc * desc, size_t width, size_t height, size_t pitch)
{
    return gnode::hooks::MainLoop::instance().cudaBindTexture2D(offset, texref, devPtr, desc, width, height, pitch);
}

cudaError_t cudaBindTextureToArray(const struct textureReference * texref, cudaArray_const_t array, const struct cudaChannelFormatDesc * desc)
{
    return gnode::hooks::MainLoop::instance().cudaBindTextureToArray(texref, array, desc);
}

cudaError_t cudaBindTextureToMipmappedArray(const struct textureReference * texref, cudaMipmappedArray_const_t mipmappedArray, const struct cudaChannelFormatDesc * desc)
{
    return gnode::hooks::MainLoop::instance().cudaBindTextureToMipmappedArray(texref, mipmappedArray, desc);
}

cudaError_t cudaUnbindTexture(const struct textureReference * texref)
{
    return gnode::hooks::MainLoop::instance().cudaUnbindTexture(texref);
}

cudaError_t cudaGetTextureAlignmentOffset(size_t * offset, const struct textureReference * texref)
{
    return gnode::hooks::MainLoop::instance().cudaGetTextureAlignmentOffset(offset, texref);
}

cudaError_t cudaGetTextureReference(const struct textureReference ** texref, const void * symbol)
{
    return gnode::hooks::MainLoop::instance().cudaGetTextureReference(texref, symbol);
}

cudaError_t cudaBindSurfaceToArray(const struct surfaceReference * surfref, cudaArray_const_t array, const struct cudaChannelFormatDesc * desc)
{
    return gnode::hooks::MainLoop::instance().cudaBindSurfaceToArray(surfref, array, desc);
}

cudaError_t cudaGetSurfaceReference(const struct surfaceReference ** surfref, const void * symbol)
{
    return gnode::hooks::MainLoop::instance().cudaGetSurfaceReference(surfref, symbol);
}

cudaError_t cudaCreateTextureObject(cudaTextureObject_t * pTexObject, const struct cudaResourceDesc * pResDesc, const struct cudaTextureDesc * pTexDesc, const struct cudaResourceViewDesc * pResViewDesc)
{
    return gnode::hooks::MainLoop::instance().cudaCreateTextureObject(pTexObject, pResDesc, pTexDesc, pResViewDesc);
}

cudaError_t cudaDestroyTextureObject(cudaTextureObject_t texObject)
{
    return gnode::hooks::MainLoop::instance().cudaDestroyTextureObject(texObject);
}

cudaError_t cudaGetTextureObjectResourceDesc(struct cudaResourceDesc * pResDesc, cudaTextureObject_t texObject)
{
    return gnode::hooks::MainLoop::instance().cudaGetTextureObjectResourceDesc(pResDesc, texObject);
}

cudaError_t cudaGetTextureObjectTextureDesc(struct cudaTextureDesc * pTexDesc, cudaTextureObject_t texObject)
{
    return gnode::hooks::MainLoop::instance().cudaGetTextureObjectTextureDesc(pTexDesc, texObject);
}

cudaError_t cudaGetTextureObjectResourceViewDesc(struct cudaResourceViewDesc * pResViewDesc, cudaTextureObject_t texObject)
{
    return gnode::hooks::MainLoop::instance().cudaGetTextureObjectResourceViewDesc(pResViewDesc, texObject);
}

cudaError_t cudaCreateSurfaceObject(cudaSurfaceObject_t * pSurfObject, const struct cudaResourceDesc * pResDesc)
{
    return gnode::hooks::MainLoop::instance().cudaCreateSurfaceObject(pSurfObject, pResDesc);
}

cudaError_t cudaDestroySurfaceObject(cudaSurfaceObject_t surfObject)
{
    return gnode::hooks::MainLoop::instance().cudaDestroySurfaceObject(surfObject);
}

cudaError_t cudaGetSurfaceObjectResourceDesc(struct cudaResourceDesc * pResDesc, cudaSurfaceObject_t surfObject)
{
    return gnode::hooks::MainLoop::instance().cudaGetSurfaceObjectResourceDesc(pResDesc, surfObject);
}

cudaError_t cudaDriverGetVersion(int * driverVersion)
{
    return gnode::hooks::MainLoop::instance().cudaDriverGetVersion(driverVersion);
}

cudaError_t cudaRuntimeGetVersion(int * runtimeVersion)
{
    return gnode::hooks::MainLoop::instance().cudaRuntimeGetVersion(runtimeVersion);
}

cudaError_t cudaGetExportTable(const void ** ppExportTable, const cudaUUID_t * pExportTableId)
{
    return gnode::hooks::MainLoop::instance().cudaGetExportTable(ppExportTable, pExportTableId);
}
}

