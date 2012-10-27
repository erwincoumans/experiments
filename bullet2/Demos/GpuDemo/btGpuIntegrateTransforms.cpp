
#include "btGpuIntegrateTransforms.h"
#include "../../../opencl/basic_initialize/btOpenCLUtils.h"

#define MSTRINGIFY(A) #A
static char* interopKernelString = 
#include "../../../opencl/broadphase_benchmark/integrateKernel.cl"

#define INTEROPKERNEL_SRC_PATH "../../../opencl/broadphase_benchmark/integrateKernel.cl"
	
btGpuIntegrateTransforms::btGpuIntegrateTransforms(cl_context ctx,cl_device_id device, cl_command_queue  q ) :
m_openclContext(ctx),
m_openclDevice(device),
m_commandQueue(q)
{
	m_prog = btOpenCLUtils::compileCLProgramFromString(ctx,device,interopKernelString,0,"",INTEROPKERNEL_SRC_PATH);
	m_integrateTransformsKernel = btOpenCLUtils::compileCLKernelFromString(ctx, device,interopKernelString, "integrateTransformsKernel" ,0,m_prog);
}

btGpuIntegrateTransforms::~btGpuIntegrateTransforms()
{

}

void btGpuIntegrateTransforms::integrate(btScalar timeStep, int numObjects,int offset, cl_mem bodyBuffer, cl_mem lv, cl_mem av, cl_mem btimes)
{
	cl_int ciErrNum = 0;

	ciErrNum = clSetKernelArg(m_integrateTransformsKernel, 0, sizeof(int), &offset);
	ciErrNum = clSetKernelArg(m_integrateTransformsKernel, 1, sizeof(int), &numObjects);
	ciErrNum = clSetKernelArg(m_integrateTransformsKernel, 2, sizeof(cl_mem), (void*)&bodyBuffer);
	
	ciErrNum = clSetKernelArg(m_integrateTransformsKernel, 3, sizeof(cl_mem), (void*)&lv);
	ciErrNum = clSetKernelArg(m_integrateTransformsKernel, 4, sizeof(cl_mem), (void*)&av);
	ciErrNum = clSetKernelArg(m_integrateTransformsKernel, 5, sizeof(cl_mem), (void*)&btimes);
					

	size_t workGroupSize = 64;
	size_t	numWorkItems = workGroupSize*((numObjects + (workGroupSize)) / workGroupSize);
				
	if (workGroupSize>numWorkItems)
		workGroupSize=numWorkItems;

	ciErrNum = clEnqueueNDRangeKernel(m_commandQueue, m_integrateTransformsKernel, 1, NULL, &numWorkItems, &workGroupSize,0 ,0 ,0);
	oclCHECKERROR(ciErrNum, CL_SUCCESS);
}