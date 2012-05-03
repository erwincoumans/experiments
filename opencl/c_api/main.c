#include "../basic_initialize/btOpenCLUtils.h"


/*
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2011 Advanced Micro Devices, Inc.  http://bulletphysics.org

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

///original author: Erwin Coumans



#include <stdio.h>
#include "../vector_add/VectorAddKernels.h"
#ifdef _WIN32
#include "btbDeviceCL.h"
#endif //_WIN32

#include "stdlib.h"
#include "btcFindPairs.h"

int ciErrNum = 0;

/*
btb //base
btm //math
btu //util
btc //collisiojn
bta //animation
btp //physics
btr //rendering
*/

int	test_RadixSort(cl_context ctx, cl_command_queue queue, cl_device_id dev)
{
	int success = 1;
	btbDevice clDevice;
	btbBuffer sortData;
	int numElements,i;
	//cl_kernel kernel;
	btbRadixSort s;
	btbSortData2ui* sortDataHost =0;

	//kernel = btOpenCLUtils_compileCLKernelFromString( ctx, dev, vectorAddCL, "VectorAdd", &ciErrNum, 0,"");

	//create 'device'?

	clDevice = btbCreateDeviceCL(ctx,dev, queue);

	//btbTestPrimitives(clDevice);

	//create buffers
	numElements = 4;//1024*12;
	sortData = btbCreateSortDataBuffer(clDevice, numElements);
	s = btbCreateRadixSort(clDevice,numElements);
	//compute
	sortDataHost = (btbSortData2ui*)malloc (sizeof(btbSortData2ui)*numElements);
	for (i=0;i<numElements;i++)
	{
		sortDataHost[i].m_key = numElements-i;
		sortDataHost[i].m_value= numElements-i;
	}
	
	btbCopyHostToBuffer(sortData, sortDataHost, numElements);
		
	btbSort(s,sortData,numElements);
	
	//get results
	btbCopyBufferToHost(sortDataHost, sortData, numElements);

	btbWaitForCompletion(clDevice);
	
	//compare results
	for (i=0;i<numElements-1;i++)
	{
		if (sortDataHost[i].m_value > sortDataHost[i+1].m_value)
		{
			success = 0;
		}
	}

	free(sortDataHost);	
	btbReleaseBuffer(sortData);
	btbReleaseDevice(clDevice);

	//clReleaseKernel(kernel);
	return success;
}



#define MAX_NUM_PARTS_IN_BITS 10

int main(int argc, char* argv[])
{
	
	cl_context			g_cxMainContext=0;
    cl_command_queue	g_cqCommandQueue=0;
    cl_device_id		g_device=0;

	cl_device_type deviceType = CL_DEVICE_TYPE_ALL;
	const char* vendorSDK = btOpenCLUtils_getSdkVendorName();
	int numPlatforms,i;
	void* glCtx=0;
	void* glDC = 0;
	int numDev =0;
	
    unsigned x = 0;
    unsigned y;
    y = (~(x&0))<<(31-MAX_NUM_PARTS_IN_BITS);
    
    
        // Get only the lower bits where the triangle index is stored
//      return (m_escapeIndexOrTriangleIndex&~((~(x&0))<<(31-MAX_NUM_PARTS_IN_BITS)));
    

	printf("This program was compiled using the %s OpenCL SDK\n",vendorSDK);
	
	numPlatforms = btOpenCLUtils_getNumPlatforms(&ciErrNum);
	printf("Num Platforms = %d\n", numPlatforms);

	

	if (0)
	for (i=0;i<numPlatforms;i++)
	{
		cl_platform_id platform = btOpenCLUtils_getPlatform(i,&ciErrNum);
		int numDevices,j;
		cl_context context;
		printf("================================\n");
		printf("Platform %d:\n", i);
		printf("================================\n");
		btOpenCLUtils_printPlatformInfo(platform);

		context = btOpenCLUtils_createContextFromPlatform(platform,deviceType,&ciErrNum,0,0,-1,-1);
		
		numDevices = btOpenCLUtils_getNumDevices(context);
		printf("Num Devices = %d\n", numDevices);
		for (j=0;j<numDevices;j++)
		{
			cl_device_id dev = btOpenCLUtils_getDevice(context,j);
			printf("--------------------------------\n");
			printf("Device %d:\n", j);
			printf("--------------------------------\n");
			btOpenCLUtils_printDeviceInfo(dev);
		}

		clReleaseContext(context);
	}

	///Easier method to initialize OpenCL using createContextFromType for a GPU
	//deviceType = CL_DEVICE_TYPE_GPU;
	deviceType = CL_DEVICE_TYPE_GPU;
	

	printf("Initialize OpenCL using btOpenCLUtils_createContextFromType for CL_DEVICE_TYPE_GPU\n");
	g_cxMainContext = btOpenCLUtils_createContextFromType(deviceType, &ciErrNum, glCtx, glDC,1,1);
	oclCHECKERROR(ciErrNum, CL_SUCCESS);

	numDev = btOpenCLUtils_getNumDevices(g_cxMainContext);

	for (i=0;i<numDev;i++)
	{
		int result;
	

		g_device = btOpenCLUtils_getDevice(g_cxMainContext,i);
		btOpenCLUtils_printDeviceInfo(g_device);
		// create a command-queue
		g_cqCommandQueue = clCreateCommandQueue(g_cxMainContext, g_device, 0, &ciErrNum);
		oclCHECKERROR(ciErrNum, CL_SUCCESS);
		//normally you would create and execute kernels using this command queue


		
		if (test_RadixSort(g_cxMainContext,g_cqCommandQueue,g_device))
		{
			printf("----------------------\n");
			printf("sorting successful\n");
		} else
		{
		printf("----------------------\n");
			printf("sorting failed\n");
		}
	
		result = testFindPairs();

		
		clReleaseCommandQueue(g_cqCommandQueue);
	}

	clReleaseContext(g_cxMainContext);
		
	return 0;
}