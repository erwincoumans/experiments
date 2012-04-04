#ifndef BTB_DEVICE_CL_H
#define BTB_DEVICE_CL_H

#include "btbPlatformDefinitions.h"

#include "../basic_initialize/btOpenCLInclude.h"

#ifdef __cplusplus
extern "C" {
#endif//__cplusplus

btbDevice btbCreateDeviceCL(cl_context ctx, cl_device_id dev, cl_command_queue q);
void btbReleaseDevice(btbDevice d);

cl_int btbGetLastErrorCL(btbDevice d);
void btbWaitForCompletion(btbDevice d);

btbBuffer btbCreateSortDataBuffer(btbDevice d, int numElements);
void btbReleaseBuffer(btbBuffer b);

int btbGetElementSizeInBytes(btbBuffer buffer);

///like memcpy destination comes first
void btbCopyHostToBuffer(btbBuffer devDest, const btbSortData2ui* hostSrc, int sizeInElements);
void btbCopyBufferToHost(btbSortData2ui* hostDest, const btbBuffer devSrc, int sizeInElements);
void btbCopyBuffer(btbBuffer dst, const btbBuffer src, int sizeInElements);

btbRadixSort btbCreateRadixSort(btbDevice d, int maxNumElements);
void btbSort(btbRadixSort s, btbBuffer buf, int numElements);

#ifdef __cplusplus
}
#endif//__cplusplus

#endif //BTB_DEVICE_CL_H
