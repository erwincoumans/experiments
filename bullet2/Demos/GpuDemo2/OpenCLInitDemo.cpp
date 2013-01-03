#include "OpenCLInitDemo.h"
#include <stdio.h>

#include "opencl/basic_initialize/btOpenCLUtils.h"


DemoBase* OpenCLInitDemo::CreateFunc()
{
	DemoBase* demo = new OpenCLInitDemo();
	return demo;
}


void OpenCLInitDemo::step()
{
	printf("OpenCLInitDemo::step\n");
	cl_int errNum;
	cl_context ctx = btOpenCLUtils::createContextFromType(CL_DEVICE_TYPE_GPU,&errNum);
	if (ctx)
	{
		cl_device_id dev = btOpenCLUtils::getDevice(ctx,0);
		if (dev)
		{
			btOpenCLUtils::printDeviceInfo(dev);
			
			clReleaseDevice(dev);
		}
	}
	clReleaseContext(ctx);
};
