
#ifndef BT_GPU_INTEGRATE_TRANSFORMS_H
#define BT_GPU_INTEGRATE_TRANSFORMS_H

#include "../../../opencl/basic_initialize/btOpenCLInclude.h"
#include "LinearMath/btScalar.h"

class btGpuIntegrateTransforms
{
protected:

	cl_context m_openclContext;
	cl_device_id m_openclDevice;
	cl_command_queue m_commandQueue;

	cl_program	m_prog;
	cl_kernel	m_integrateTransformsKernel;

public:

	btGpuIntegrateTransforms(cl_context ctx,cl_device_id device, cl_command_queue  q );

	virtual ~btGpuIntegrateTransforms();

	void integrate(btScalar timeStep, int numObjects,int offset, cl_mem bodyBuffer, cl_mem lv, cl_mem av, cl_mem btimes);

};

#endif //BT_GPU_INTEGRATE_TRANSFORMS_H