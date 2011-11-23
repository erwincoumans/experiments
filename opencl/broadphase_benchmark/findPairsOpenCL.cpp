
#include "findPairsOpenCL.h"
#include "../basic_initialize/btOpenCLUtils.h"

#define MSTRINGIFY(A) #A
static char* broadphaseKernelString = 
#include "broadphaseKernel.cl"

static const char* sp3dGridBroadphaseSource = 
#include "../3dGridBroadphase/Shared/bt3dGridBroadphaseOCL.cl"




void initFindPairs(btFindPairsIO& fpio,cl_context cxMainContext, cl_device_id device, cl_command_queue commandQueue, int maxHandles, int maxPairsPerBody)
{

	//m_proxies.push_back( proxy );

	fpio.m_mainContext = cxMainContext;
	fpio.m_cqCommandQue = commandQueue;
	fpio.m_device = device;
	fpio.m_broadphaseBruteForceKernel = btOpenCLUtils::compileCLKernelFromString(cxMainContext,device, broadphaseKernelString, "broadphaseKernel" );
	fpio.m_broadphaseGridKernel = btOpenCLUtils::compileCLKernelFromString(cxMainContext,device, broadphaseKernelString, "broadphaseGridKernel" );
	fpio.m_broadphaseColorKernel = btOpenCLUtils::compileCLKernelFromString(cxMainContext,device, broadphaseKernelString, "broadphaseColorKernel" );

	



}

void	findPairsOpenCLBruteForce(btFindPairsIO& fpio)
{

			int ciErrNum = 0;

			int numObjects = fpio.m_numObjects;
			int offset = fpio.m_positionOffset;

			ciErrNum = clSetKernelArg(fpio.m_broadphaseBruteForceKernel, 0, sizeof(int), &offset);
			ciErrNum = clSetKernelArg(fpio.m_broadphaseBruteForceKernel, 1, sizeof(int), &numObjects);
			ciErrNum = clSetKernelArg(fpio.m_broadphaseBruteForceKernel, 2, sizeof(cl_mem), (void*)&fpio.m_clObjectsBuffer);
		
			size_t numWorkItems = numObjects;///workGroupSize*((NUM_OBJECTS + (workGroupSize)) / workGroupSize);
			size_t workGroupSize = 64;
			ciErrNum = clEnqueueNDRangeKernel(fpio.m_cqCommandQue, fpio.m_broadphaseBruteForceKernel, 1, NULL, &numWorkItems, &workGroupSize,0 ,0 ,0);
			oclCHECKERROR(ciErrNum, CL_SUCCESS);
}

void	findPairsOpenCL(btFindPairsIO& fpio)
{

			int ciErrNum = 0;

			int numObjects = fpio.m_numObjects;
			int offset = fpio.m_positionOffset;

			ciErrNum = clSetKernelArg(fpio.m_broadphaseGridKernel, 0, sizeof(int), &offset);
			ciErrNum = clSetKernelArg(fpio.m_broadphaseGridKernel, 1, sizeof(int), &numObjects);
			ciErrNum = clSetKernelArg(fpio.m_broadphaseGridKernel, 2, sizeof(cl_mem), (void*)&fpio.m_clObjectsBuffer);
			ciErrNum = clSetKernelArg(fpio.m_broadphaseGridKernel, 3, sizeof(cl_mem), (void*)&fpio.m_dAABB);

			size_t numWorkItems = numObjects;///workGroupSize*((NUM_OBJECTS + (workGroupSize)) / workGroupSize);
			size_t workGroupSize = 64;
			ciErrNum = clEnqueueNDRangeKernel(fpio.m_cqCommandQue, fpio.m_broadphaseGridKernel, 1, NULL, &numWorkItems, &workGroupSize,0 ,0 ,0);
			oclCHECKERROR(ciErrNum, CL_SUCCESS);
}


void	drawPairsOpenCL(btFindPairsIO&	fpio)
{
	int ciErrNum = 0;

	int numObjects = fpio.m_numObjects;
	int offset = fpio.m_positionOffset;

	ciErrNum = clSetKernelArg(fpio.m_broadphaseColorKernel, 0, sizeof(int), &offset);
	ciErrNum = clSetKernelArg(fpio.m_broadphaseColorKernel, 1, sizeof(int), &fpio.m_numObjects);
	ciErrNum = clSetKernelArg(fpio.m_broadphaseColorKernel, 2, sizeof(cl_mem), (void*)&fpio.m_clObjectsBuffer);
	ciErrNum = clSetKernelArg(fpio.m_broadphaseColorKernel, 3, sizeof(cl_mem), (void*)&fpio.m_dPairsChangedXY);
	ciErrNum = clSetKernelArg(fpio.m_broadphaseColorKernel, 4, sizeof(int), &fpio.m_numOverlap);


	size_t numWorkItems = numObjects;///workGroupSize*((NUM_OBJECTS + (workGroupSize)) / workGroupSize);
	size_t workGroupSize = 64;
	ciErrNum = clEnqueueNDRangeKernel(fpio.m_cqCommandQue, fpio.m_broadphaseColorKernel, 1, NULL, &numWorkItems, &workGroupSize,0 ,0 ,0);
	oclCHECKERROR(ciErrNum, CL_SUCCESS);
}



void releaseFindPairs(btFindPairsIO& fpio)
{
	clReleaseKernel(fpio.m_broadphaseGridKernel);
	clReleaseKernel(fpio.m_broadphaseColorKernel);
	clReleaseKernel(fpio.m_broadphaseBruteForceKernel);

}

