

#include "btbDeviceCL.h"
#include "../broadphase_benchmark/btOpenCLArray.h"
#include "../broadphase_benchmark/btRadixSort32CL.h"
#include "../broadphase_benchmark/btFillCL.h"
#include "../broadphase_benchmark/btPrefixScanCL.h"


typedef struct
{
    cl_context m_ctx;
    cl_device_id m_device;
    cl_command_queue m_queue;
} btbMyDeviceCL;

btbDevice btbCreateDeviceCL(cl_context ctx, cl_device_id dev, cl_command_queue q)
{
    btbMyDeviceCL* clDev = (btbMyDeviceCL*)malloc(sizeof(btbMyDeviceCL));
    clDev->m_ctx = ctx;
    clDev->m_device = dev;
    clDev->m_queue = q;
	return (btbDevice) clDev;
}

void btbReleaseDevice(btbDevice d)
{
	btbMyDeviceCL* dev = (btbMyDeviceCL*)d;
	btbAssert(dev);
	delete dev;
}

cl_int btbGetLastErrorCL(btbDevice d)
{
	return 0;
}

void btbWaitForCompletion(btbDevice d)
{
	btbMyDeviceCL* dev = (btbMyDeviceCL*)d;
    clFinish(dev->m_queue);
}

btbBuffer btbCreateSortDataBuffer(btbDevice d, int numElements)
{
	btbMyDeviceCL* dev = (btbMyDeviceCL*) d;
    btOpenCLArray<btSortData>* buf = new btOpenCLArray<btSortData>(dev->m_ctx,dev->m_queue);
    buf->resize(numElements);
	return (btbBuffer)buf;
}

void btbReleaseBuffer(btbBuffer b)
{
	btOpenCLArray<btSortData>* buf = (btOpenCLArray<btSortData>*) b;
	delete buf;
}

///like memcpy destination comes first
void btbCopyHostToBuffer(btbBuffer devDest, const btbSortData2ui* hostSrc, int sizeInElements)
{
    const btSortData* hostPtr = (const btSortData*)hostSrc;
	btOpenCLArray<btSortData>* devBuf = (btOpenCLArray<btSortData>*)devDest;
	devBuf->copyFromHostPointer(hostPtr,sizeInElements);
}

void btbCopyBufferToHost(btbSortData2ui* hostDest, const btbBuffer devSrc, int sizeInElements)
{
    btSortData* hostPtr = (btSortData*)hostDest;
	btOpenCLArray<btSortData>* devBuf = (btOpenCLArray<btSortData>*)devSrc;
	devBuf->copyToHostPointer(hostPtr,sizeInElements);
}


/*

void btbCopyBuffer(btbBuffer dst, const btbBuffer src, int sizeInElements)
{

}

*/

void btbTestPrimitives(btbDevice d)
{
	btbMyDeviceCL* dev = (btbMyDeviceCL*)d;
/*
	btRadixSort32CL sort(dev->m_ctx,dev->m_device, dev->m_queue);
	btOpenCLArray<btSortData> keyValuesInOut(dev->m_ctx,dev->m_queue);

	btAlignedObjectArray<btSortData> hostData;
	btSortData key; key.m_key = 2; key.m_value = 2;
	hostData.push_back(key); key.m_key = 1; key.m_value = 1;
	hostData.push_back(key);

	keyValuesInOut.copyFromHost(hostData);
	sort.execute(keyValuesInOut);
	keyValuesInOut.copyToHost(hostData);
*/

	btFillCL filler(dev->m_ctx,dev->m_device, dev->m_queue);
	btInt2 value;
	value.x = 4;
	value.y = 5;
	btOpenCLArray<btInt2> testArray(dev->m_ctx,dev->m_queue);
	testArray.resize(1024);
	filler.execute(testArray,value, testArray.size());
	btAlignedObjectArray<btInt2> hostInt2Array;
	testArray.copyToHost(hostInt2Array);


	btPrefixScanCL scan(dev->m_ctx,dev->m_device, dev->m_queue);

	unsigned int sum;
	btOpenCLArray<unsigned int>src(dev->m_ctx,dev->m_queue);

	src.resize(16);

	filler.execute(src,2, src.size());

	btAlignedObjectArray<unsigned int>hostSrc;
	src.copyToHost(hostSrc);


	btOpenCLArray<unsigned int>dest(dev->m_ctx,dev->m_queue);
	scan.execute(src,dest,src.size(),&sum);
	dest.copyToHost(hostSrc);


}

btbRadixSort btbCreateRadixSort(btbDevice d, int maxNumElements)
{
	btbMyDeviceCL* dev = (btbMyDeviceCL*)d;
    btRadixSort32CL* radix = new btRadixSort32CL(dev->m_ctx,dev->m_device,dev->m_queue);
	return (btbRadixSort)radix;
}
void btbSort(btbRadixSort s, btbBuffer b, int numElements)
{
	btRadixSort32CL* radix = (btRadixSort32CL*)s;
	btOpenCLArray<btSortData>* buf = (btOpenCLArray<btSortData>*)b;
	radix->execute( *buf);
}
