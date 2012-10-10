/*
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2003-2007 Erwin Coumans  http://continuousphysics.com/Bullet/

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

#include "BasicDemo.h"
#include "GlutStuff.h"
#include "btBulletDynamicsCommon.h"
#include "LinearMath/btHashMap.h"
#include "../basic_initialize/btOpenCLUtils.h"

cl_context g_cxMainContext;
cl_device_id g_device;
cl_command_queue g_cqCommandQueue;

void InitCL(int preferredDeviceIndex, int preferredPlatformIndex)
{
	bool useInterop = false;

	void* glCtx=0;
	void* glDC = 0;

	int ciErrNum = 0;
//#ifdef CL_PLATFORM_INTEL
//	cl_device_type deviceType = CL_DEVICE_TYPE_ALL;
//#else
	cl_device_type deviceType = CL_DEVICE_TYPE_GPU;
//#endif

	

	if (useInterop)
	{
	//	g_cxMainContext = btOpenCLUtils::createContextFromType(deviceType, &ciErrNum, glCtx, glDC);
	} else
	{
		cl_platform_id platform;
		g_cxMainContext = btOpenCLUtils::createContextFromType(deviceType, &ciErrNum, 0,0,preferredDeviceIndex, preferredPlatformIndex, &platform);
		if (g_cxMainContext && platform)
		{
			btOpenCLUtils::printPlatformInfo(platform);
		}
	}


	oclCHECKERROR(ciErrNum, CL_SUCCESS);

	int numDev = btOpenCLUtils::getNumDevices(g_cxMainContext);

	if (numDev>0)
	{
		g_device= btOpenCLUtils::getDevice(g_cxMainContext,0);
		g_cqCommandQueue = clCreateCommandQueue(g_cxMainContext, g_device, 0, &ciErrNum);
		oclCHECKERROR(ciErrNum, CL_SUCCESS);
        
		
        btOpenCLUtils::printDeviceInfo(g_device);

	}

}


	
int main(int argc,char** argv)
{
	InitCL(-1,-1);

	BasicDemo ccdDemo(g_cxMainContext,g_device,g_cqCommandQueue);
	ccdDemo.initPhysics();


#ifdef CHECK_MEMORY_LEAKS
	ccdDemo.exitPhysics();
#else
	glutmain(argc, argv,1024,600,"Bullet Physics Demo. http://bulletphysics.org",&ccdDemo);
#endif
	
//	setupGUI(1024,768);
	glutMainLoop();
	
	//default glut doesn't return from mainloop
	return 0;
}

