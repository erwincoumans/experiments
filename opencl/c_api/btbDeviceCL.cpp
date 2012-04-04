
#include "Adl/Adl.h"
#include "btbDeviceCL.h"
#include "AdlPrimitives/Sort/SortData.h"

btbDevice btbCreateDeviceCL(cl_context ctx, cl_device_id dev, cl_command_queue q)
{
	adl::DeviceCL* clDev = new adl::DeviceCL();
	clDev->m_deviceIdx = dev;
	clDev->m_context = ctx;
	clDev->m_commandQueue = q;
	clDev->m_kernelManager = new adl::KernelManager;
	
	
	return (btbDevice) clDev;
}

void btbReleaseDevice(btbDevice d)
{
	adl::Device* dev = (adl::Device*)d;
	btbAssert(dev);
	dev->release();
	delete dev;
}

cl_int btbGetLastErrorCL(btbDevice d)
{
	return 0;
}

void btbWaitForCompletion(btbDevice d)
{

}

btbBuffer btbCreateSortDataBuffer(btbDevice d, int numElements)
{
	adl::Device* dev = (adl::Device*) d;
	adl::Buffer<adl::SortData>* buf = new adl::Buffer<adl::SortData>(dev, numElements);
	return (btbBuffer)buf;

}
void btbReleaseBuffer(btbBuffer b)
{
	adl::BufferBase* buf = (adl::BufferBase*) b;
	delete buf;
}

/*

int btbGetElementSizeInBytes(btbBuffer buffer)
{

}

///like memcpy destination comes first
void btbCopyHostToBuffer(btbBuffer devDest, const char* hostSrc, int sizeInElements)
{

}
void btbCopyBufferToHost(char* hostDest, const btbBuffer devSrc, int sizeInElements)
{

}
void btbCopyBuffer(btbBuffer dst, const btbBuffer src, int sizeInElements)
{

}

btbRadixSort btbCreateRadixSort(btbDevice d, int maxNumElements)
{

}
void btbSort(btbRadixSort s, btbBuffer buf, int numElements)
{

}
*/
