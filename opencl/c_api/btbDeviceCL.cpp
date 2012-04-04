
#include "Adl/Adl.h"
#include "btbDeviceCL.h"
#include "AdlPrimitives/Sort/SortData.h"
#include "AdlPrimitives/Sort/RadixSort32.h"

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
	adl::Device* dev = (adl::Device*)d;
	dev->waitForCompletion();
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

///like memcpy destination comes first
void btbCopyHostToBuffer(btbBuffer devDest, const btbSortData2ui* hostSrc, int sizeInElements)
{
	
	adl::Buffer<adl::SortData>* devBuf = (adl::Buffer<adl::SortData>*)devDest;
	//adl::DeviceHost::allocate
	adl::DeviceHost devHost;
	adl::HostBuffer<adl::SortData> hostBuf;
	hostBuf.m_ptr = (adl::SortData*)hostSrc;
	hostBuf.m_device = &devHost;
	hostBuf.m_size = sizeInElements;
	hostBuf.m_allocated = false;
	devBuf->write(hostBuf,sizeInElements);
}
void btbCopyBufferToHost(btbSortData2ui* hostDest, const btbBuffer devSrc, int sizeInElements)
{
	adl::HostBuffer<adl::SortData> hostBuf;
	adl::DeviceHost devHost;
	hostBuf.m_ptr = (adl::SortData*)hostDest;
	hostBuf.m_size = sizeInElements;
	hostBuf.m_device = &devHost;
	hostBuf.m_allocated = false;

	adl::Buffer<adl::SortData>* devBufSrc = (adl::Buffer<adl::SortData>*)devSrc;
	devBufSrc->read(hostBuf,sizeInElements);

}


/*

void btbCopyBuffer(btbBuffer dst, const btbBuffer src, int sizeInElements)
{

}

*/

btbRadixSort btbCreateRadixSort(btbDevice d, int maxNumElements)
{
	adl::Device* dev = (adl::Device*)d;
	adl::RadixSort32<adl::TYPE_CL>::Data* dataC = adl::RadixSort32<adl::TYPE_CL>::allocate( dev, maxNumElements);
	return (btbRadixSort)dataC;
}
void btbSort(btbRadixSort s, btbBuffer b, int numElements)
{
	adl::RadixSort32<adl::TYPE_CL>::Data* dataC = (adl::RadixSort32<adl::TYPE_CL>::Data*)s;
	adl::Buffer<adl::SortData>* buf = (adl::Buffer<adl::SortData>*)b;
	adl::RadixSort32<adl::TYPE_CL>::execute( dataC, *buf, numElements);
}
