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
#ifdef __APPLE__
#include <unistd.h>
#endif

#ifdef _WIN32
#include "Windows.h"
#define sleep(x) Sleep(x*1000)
#endif

#ifndef __APPLE__
#define MAX_KERNEL_TESTS 4
#else
#define MAX_KERNEL_TESTS 3
#endif


#include "../broadphase_benchmark/btLauncherCL.h"

#include "../basic_initialize/btOpenCLUtils.h"

#include "../broadphase_benchmark/sapFastKernels.h"
#include "../broadphase_benchmark/sapKernels.h"
#include "../../dynamics/basic_demo/Stubs/SolverKernels.h"
#include "../../opencl/gpu_rigidbody_pipeline2/satClipKernels.h"


#define SOLVER_KERNEL_PATH "../../dynamics/basic_demo/Stubs/SolverKernels.cl"

#define CLIP_HULL_KERNEL_PATH "../../opencl/gpu_rigidbody_pipeline2/satClipHullContacts.cl"

cl_context			g_cxMainContext;
cl_command_queue	g_cqCommandQue;
cl_device_id        g_device;




void InitCL(int preferredDeviceIndex, int preferredPlatformIndex, bool useInterop)
{
	void* glCtx=0;
	void* glDC = 0;
    
#ifdef _WIN32
	
#else //!_WIN32
#ifndef __APPLE__
    GLXContext glCtx = glXGetCurrentContext();
    glDC = wglGetCurrentDC();//??
#endif
#endif //!_WIN32
    
    
	int ciErrNum = 0;
    //#ifdef CL_PLATFORM_INTEL
    //	cl_device_type deviceType = CL_DEVICE_TYPE_ALL;
    //#else
	cl_device_type deviceType = CL_DEVICE_TYPE_GPU;
    //#endif
    
	
    
	if (useInterop)
	{
		g_cxMainContext = btOpenCLUtils::createContextFromType(deviceType, &ciErrNum, glCtx, glDC);
	} else
	{
		g_cxMainContext = btOpenCLUtils::createContextFromType(deviceType, &ciErrNum, 0,0,preferredDeviceIndex, preferredPlatformIndex);
	}
    
    
	oclCHECKERROR(ciErrNum, CL_SUCCESS);
    
	int numDev = btOpenCLUtils::getNumDevices(g_cxMainContext);
    
	if (numDev>0)
	{
		g_device= btOpenCLUtils::getDevice(g_cxMainContext,0);
		g_cqCommandQue = clCreateCommandQueue(g_cxMainContext, g_device, 0, &ciErrNum);
		oclCHECKERROR(ciErrNum, CL_SUCCESS);
        
        btOpenCLUtils::printDeviceInfo(g_device);
        
	}
    
}





int testSapKernel_computePairsKernelOriginal(int kernelIndex)
{
           
    const char* sapSrc = sapCL;
    const char* sapFastSrc = sapFastCL;
    

    
        cl_int errNum=0;
        
        cl_program sapProg = btOpenCLUtils::compileCLProgramFromString(g_cxMainContext,g_device,sapSrc,&errNum,"","../../opencl/broadphase_benchmark/sap.cl");
        btAssert(errNum==CL_SUCCESS);
#ifndef __APPLE__
    cl_program sapFastProg = btOpenCLUtils::compileCLProgramFromString(g_cxMainContext,g_device,sapFastSrc,&errNum,"","../../opencl/broadphase_benchmark/sapFast.cl");
    btAssert(errNum==CL_SUCCESS);
#endif
    
        cl_kernel m_sapKernel = 0;
        
        switch (kernelIndex)
        {
            case 0:
                m_sapKernel = btOpenCLUtils::compileCLKernelFromString(g_cxMainContext, g_device,sapSrc, "computePairsKernelOriginal",&errNum,sapProg );
                break;
            case 1:
                m_sapKernel = btOpenCLUtils::compileCLKernelFromString(g_cxMainContext, g_device,sapSrc, "computePairsKernelBarrier",&errNum,sapProg );
                break;
            case 2:
                m_sapKernel = btOpenCLUtils::compileCLKernelFromString(g_cxMainContext, g_device,sapSrc, "computePairsKernelLocalSharedMemory",&errNum,sapProg );
                break;
#ifndef __APPLE__
            case 3:
                m_sapKernel = btOpenCLUtils::compileCLKernelFromString(g_cxMainContext, g_device,sapFastSrc, "computePairsKernel",&errNum,sapFastProg );
                break;
#endif
                
            default:
            {
                assert(0);
            }
        }
        
        btAssert(errNum==CL_SUCCESS);
        
        

        btLauncherCL launcher(g_cqCommandQue, m_sapKernel);
        const char* fileName = "m_sapKernelArgs.bin";
        FILE* f = fopen(fileName,"rb");
        if (f)
        {
            int sizeInBytes=0;
            if (fseek(f, 0, SEEK_END) || (sizeInBytes = ftell(f)) == EOF || fseek(f, 0, SEEK_SET)) 
            {
                printf("error, cannot get file size\n");
                exit(0);
            }
            
            unsigned char* buf = (unsigned char*) malloc(sizeInBytes);
            fread(buf,sizeInBytes,1,f);
            int serializedBytes = launcher.deserializeArgs(buf, sizeInBytes,g_cxMainContext);
            int num = *(int*)&buf[serializedBytes];

            launcher.launch1D( num);
            btOpenCLArray<int> pairCount(g_cxMainContext, g_cqCommandQue);
            int numElements = launcher.m_arrays[2]->size()/sizeof(int);
            pairCount.setFromOpenCLBuffer(launcher.m_arrays[2]->getBufferCL(),numElements);
            int count = pairCount.at(0);
            printf("overlapping pairs = %d\n",count);
           
            
        } else {
            printf("error: cannot find file %s\n",fileName);
        }

        
    clFinish(g_cqCommandQue);
        
    clReleaseKernel(m_sapKernel);
    clReleaseProgram(sapProg);
    clFinish(g_cqCommandQue);
       
}


void testSolverKernel()
{
	const char* solverKernelSource = solverKernelsCL;
	cl_int pErrNum=0;
	const char* additionalMacros = "";

	cl_program solverProg= btOpenCLUtils::compileCLProgramFromString( g_cxMainContext, g_device, solverKernelSource, &pErrNum,additionalMacros, SOLVER_KERNEL_PATH);
	btAssert(solverProg);
		
	cl_kernel m_batchSolveKernel= btOpenCLUtils::compileCLKernelFromString( g_cxMainContext, g_device, solverKernelSource, "BatchSolveKernel", &pErrNum, solverProg,additionalMacros );
	btAssert(m_batchSolveKernel);
	

	 btLauncherCL launcher(g_cqCommandQue, m_batchSolveKernel);
        const char* fileName = "m_batchSolveKernel.bin";
        FILE* f = fopen(fileName,"rb");
        if (f)
        {
            int sizeInBytes=0;
            if (fseek(f, 0, SEEK_END) || (sizeInBytes = ftell(f)) == EOF || fseek(f, 0, SEEK_SET)) 
            {
                printf("error, cannot get file size\n");
                exit(0);
            }
            
            unsigned char* buf = (unsigned char*) malloc(sizeInBytes);
            fread(buf,sizeInBytes,1,f);
            int serializedBytes = launcher.deserializeArgs(buf, sizeInBytes,g_cxMainContext);
            int num = *(int*)&buf[serializedBytes];

            launcher.launch1D( num);

            /*btOpenCLArray<int> pairCount(g_cxMainContext, g_cqCommandQue);
            int numElements = launcher.m_arrays[2]->size()/sizeof(int);
            pairCount.setFromOpenCLBuffer(launcher.m_arrays[2]->getBufferCL(),numElements);
            int count = pairCount.at(0);
            printf("overlapping pairs = %d\n",count);
           */

            
        } else {
            printf("error: cannot find file %s\n",fileName);
        }

        
    clFinish(g_cqCommandQue);
        
    clReleaseKernel(m_batchSolveKernel);
    clReleaseProgram(solverProg);
    clFinish(g_cqCommandQue);

}


void testCLipHullKernel()
{

	const char* clipHullKernelSource = satClipKernelsCL;
	cl_int pErrNum=0;
	const char* additionalMacros = "";

	cl_program clipHullProg= btOpenCLUtils::compileCLProgramFromString( g_cxMainContext, g_device, clipHullKernelSource, &pErrNum,additionalMacros, CLIP_HULL_KERNEL_PATH);
	btAssert(clipHullProg);
		
	cl_kernel m_clipHullKernel= btOpenCLUtils::compileCLKernelFromString( g_cxMainContext, g_device, clipHullKernelSource, "clipFacesAndContactReductionKernel", &pErrNum, clipHullProg,additionalMacros );
	btAssert(m_clipHullKernel);
	

	 btLauncherCL launcher(g_cqCommandQue, m_clipHullKernel);
        const char* fileName = "clipFacesAndContactReductionKernel_before.bin";
        FILE* fin = fopen(fileName,"rb");
        if (fin)
        {
            int sizeInBytes=0;
            if (fseek(fin, 0, SEEK_END) || (sizeInBytes = ftell(fin)) == EOF || fseek(fin, 0, SEEK_SET)) 
            {
                printf("error, cannot get file size\n");
                exit(0);
            }
            
            unsigned char* bufIn = (unsigned char*) malloc(sizeInBytes);
            fread(bufIn,sizeInBytes,1,fin);
            int serializedBytes = launcher.deserializeArgs(bufIn, sizeInBytes,g_cxMainContext);
            int num = *(int*)&bufIn[serializedBytes];
			fclose(fin);
			fin=0;

			launcher.serializeToFile("testbefore.bin",1);

            launcher.launch1D( num);
			clFinish(g_cqCommandQue);

			launcher.serializeToFile("testafter.bin",1);



			//verify with precomputed (gold) output binary
			const char* fileNameGold = "clipFacesAndContactReductionKernel_after.bin";
			FILE* fg = fopen(fileNameGold,"rb");
			if (fg)
			{
				int sizeInBytes=0;
				if (fseek(fg, 0, SEEK_END) || (sizeInBytes = ftell(fg)) == EOF || fseek(fg, 0, SEEK_SET)) 
				{
					printf("error, cannot get file size\n");
					exit(0);
				}
            
				unsigned char* bufgold = (unsigned char*) malloc(sizeInBytes);
				fread(bufgold,sizeInBytes,1,fg);
				unsigned char* hitdata = &bufgold[0x64a240];
				fclose(fg);
				fg=0;

				int result = launcher.validateResults(bufgold,sizeInBytes,g_cxMainContext);
				if (result>=0)
				{
					printf("computation results OK\n");
				}
				else
				{
					printf("test failed with output %d\n", result);
				}
			}
			
            /*btOpenCLArray<int> pairCount(g_cxMainContext, g_cqCommandQue);
            int numElements = launcher.m_arrays[2]->size()/sizeof(int);
            pairCount.setFromOpenCLBuffer(launcher.m_arrays[2]->getBufferCL(),numElements);
            int count = pairCount.at(0);
            printf("overlapping pairs = %d\n",count);
           */

            
        } else {
            printf("error: cannot find file %s\n",fileName);
        }

        
    clFinish(g_cqCommandQue);
        
    clReleaseKernel(m_clipHullKernel);
    clReleaseProgram(clipHullProg);
    clFinish(g_cqCommandQue);

}




int actualMain(int argc, char* argv[])
{
	int ciErrNum = 0;
  
    bool interop = false;
    InitCL(-1,-1,interop);
    
    testCLipHullKernel();

/*    for (int i=0;i<MAX_KERNEL_TESTS;i++)
        testSapKernel_computePairsKernelOriginal(i);
	
    testSolverKernel();
	*/


    
    //int numPairs = pairCount.at(0);
    
    
    clReleaseCommandQueue(g_cqCommandQue);
	clReleaseContext(g_cxMainContext);

    printf("the end\n");
	return 0;
    
    
    
}


int main(int argc, char* argv[])
{
    actualMain(argc,argv);  
#ifdef __APPLE__
	//use a sleep for the 'Leaks' app to work
	sleep(2);
#endif

    printf("finished\n");
}


