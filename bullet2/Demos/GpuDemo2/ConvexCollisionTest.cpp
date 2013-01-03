#include "ConvexCollisionTest.h"
#include <stdio.h>

#include "opencl/gpu_rigidbody_pipeline2/ConvexHullContact.h"
#include "opencl/basic_initialize/btOpenCLUtils.h"


DemoBase* ConvexCollisionTest::CreateFunc()
{
	DemoBase* demo = new ConvexCollisionTest();
	return demo;
}

void ConvexCollisionTest::step()
{
	printf("ConvexCollisionTest::step\n");
	
	cl_int errNum;
	cl_context ctx = btOpenCLUtils::createContextFromType(CL_DEVICE_TYPE_GPU,&errNum);
	assert (ctx);
	cl_device_id dev = btOpenCLUtils::getDevice(ctx,0);
	assert(dev);
	btOpenCLUtils::printDeviceInfo(dev);
	
	cl_command_queue queue = clCreateCommandQueue(ctx, dev, 0, &errNum);
	
	
	GpuSatCollision* sat = new GpuSatCollision(ctx,dev,queue);
	
	int bodyIndexA;
	int bodyIndexB;
	int collidableIndexA;
	int collidableIndexB;
	const btAlignedObjectArray<RigidBodyBase::Body> *bodyBuf;
	btAlignedObjectArray<Contact4> *contactOut;
	int nContacts;
	btAlignedObjectArray<ConvexPolyhedronCL> hostConvexDataA;
	btAlignedObjectArray<ConvexPolyhedronCL> gpuConvexDataB;
	btAlignedObjectArray<btVector3> verticesA;
	btAlignedObjectArray<btVector3> uniqueEdgesA;
	btAlignedObjectArray<btGpuFace> facesA;
	btAlignedObjectArray<int> indicesA;
	btAlignedObjectArray<btVector3> gpuVerticesB;
	btAlignedObjectArray<btVector3> gpuUniqueEdgesB;
	btAlignedObjectArray<btGpuFace> gpuFacesB;
	btAlignedObjectArray<int> gpuIndicesB;
	btAlignedObjectArray<btCollidable> hostCollidablesA;
	btAlignedObjectArray<btCollidable> gpuCollidablesB;
	
	
	//sat->computeConvexConvexContactsGPUSATSingle();
	
	delete sat;
	
	clReleaseCommandQueue(queue);
	clReleaseDevice(dev);
	clReleaseContext(ctx);
	
}
