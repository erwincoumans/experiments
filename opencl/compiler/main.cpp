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

#include "btOpenCLUtils.h"
#include <stdio.h>
#include <stdlib.h>
#include "../../opencl/gpu_rigidbody_pipeline/CommandLineArgs.h"

cl_context			g_cxMainContext;
cl_command_queue	g_cqCommandQue;


void Usage()
{
	printf("program.exe --filename=\"filename.cl\" --verbose=<0,1>");
}

int main(int argc, char* argv[])
{
	int ciErrNum = 0;
	char* fileName = "sap.cl";
	int verbose = 0;
	CommandLineArgs args(argc,argv);


	args.GetCmdLineArgument("filename", fileName);
	args.GetCmdLineArgument("verbose", verbose);

	if (!fileName || args.CheckCmdLineFlag("help"))
	{
		Usage();
		return 0;
	}


	if (verbose)
	{
		printf("verbose = %d\n",verbose);
		printf("inputfile=%s\n",fileName);
	}

	cl_device_type deviceType = CL_DEVICE_TYPE_ALL;
	const char* vendorSDK = btOpenCLUtils::getSdkVendorName();
	if (verbose)
		printf("This program was compiled using the %s OpenCL SDK\n",vendorSDK);
	int numPlatforms = btOpenCLUtils::getNumPlatforms();
	if (verbose)
		printf("Num Platforms = %d\n", numPlatforms);

	///Easier method to initialize OpenCL using createContextFromType for a GPU
	deviceType = CL_DEVICE_TYPE_GPU;

	void* glCtx=0;
	void* glDC = 0;
	if (verbose)
		printf("Initialize OpenCL using btOpenCLUtils::createContextFromType for CL_DEVICE_TYPE_GPU\n");
	g_cxMainContext = btOpenCLUtils::createContextFromType(deviceType, &ciErrNum, glCtx, glDC);
	oclCHECKERROR(ciErrNum, CL_SUCCESS);

	int numDev = btOpenCLUtils::getNumDevices(g_cxMainContext);
	int preferredDeviceIndex=0;

	cl_device_id		device;
	device = btOpenCLUtils::getDevice(g_cxMainContext,preferredDeviceIndex);
	btOpenCLDeviceInfo clInfo;
	btOpenCLUtils::getDeviceInfo(device,&clInfo);
	if (verbose)
		btOpenCLUtils::printDeviceInfo(device);
	// create a command-queue
	g_cqCommandQue = clCreateCommandQueue(g_cxMainContext, device, 0, &ciErrNum);
	oclCHECKERROR(ciErrNum, CL_SUCCESS);

	char* kernelSource = 0;

	FILE* f = fopen(fileName,"rb");
	if (f)
	{
		fseek(f, 0L, SEEK_END);
		int mFileLen = ftell(f);
		fseek(f, 0L, SEEK_SET);

		kernelSource = (char*)malloc(mFileLen+1);

		int actualBytesRead = fread(kernelSource,1,mFileLen,f);
		kernelSource[mFileLen]=0;

		fclose(f);

		cl_int errNum=0;
		const char* additionalMacros ="";
		const char* srcFileNameForCaching=fileName;

		cl_program prog = btOpenCLUtils::compileCLProgramFromString( g_cxMainContext,device, kernelSource, &errNum,
													additionalMacros, 0);//srcFileNameForCaching);

		if (prog)
		{
			cl_uint num_kernels = 1024;
			cl_kernel kernelArray[1024];
			cl_uint num_kernels_ret;

			errNum = clCreateKernelsInProgram (prog,
								num_kernels,
							  kernelArray,
							&num_kernels_ret);
			if (errNum==CL_SUCCESS)
			{
				printf("clCreateKernelsInProgram OK\n");
			} else
			{
				printf("ERROR clCreateKernelsInProgram failed\n");
			}
		} else
		{
			printf("ERROR compiling %s failed\n", fileName);
		}
	} else
	{
		printf("ERROR FILE not found %s\n", fileName);
	}
	clReleaseCommandQueue(g_cqCommandQue);

	clReleaseContext(g_cxMainContext);

	return 0;
}
