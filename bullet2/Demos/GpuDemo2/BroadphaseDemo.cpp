#include "BroadphaseDemo.h"

#include "opencl/basic_initialize/btOpenCLUtils.h"
#include "opencl/gpu_rigidbody_pipeline2/btGpuSapBroadphase.h"
#include "LinearMath/btVector3.h"


DemoBase* BroadphaseDemo::CreateFunc()
{
	DemoBase* demo = new BroadphaseDemo();
	return demo;
}

void BroadphaseDemo::step()
{
	cl_int errNum;
	cl_context ctx = btOpenCLUtils::createContextFromType(CL_DEVICE_TYPE_GPU,&errNum);
	assert (ctx);
	cl_device_id dev = btOpenCLUtils::getDevice(ctx,0);
	assert(dev);
	btOpenCLUtils::printDeviceInfo(dev);

	cl_command_queue queue = clCreateCommandQueue(ctx, dev, 0, &errNum);
	
	
	btGpuSapBroadphase* bp = new btGpuSapBroadphase(ctx ,dev,queue);//
	
	btVector3 aabbMin(2,0,0);
	btVector3 aabbMax(3,1,1);
	int userPtr=2;
	int userPtr2 = 3;
	
	bp->createProxy(aabbMin,aabbMax,userPtr,0,0);
	bp->createProxy(aabbMin,aabbMax,userPtr2,0,0);

	bp->writeAabbsToGpu();
	
	bool forceHost = false;
	bp->calculateOverlappingPairs(forceHost);
	int numOverlap = bp->getNumOverlap();
	cl_mem overlap = bp->getOverlappingPairBuffer();
	btAlignedObjectArray<btInt2> overlapCpu;
	btOpenCLArray<btInt2> overlapGpu(ctx,queue);
	overlapGpu.setFromOpenCLBuffer(overlap,numOverlap);
	overlapGpu.copyToHost(overlapCpu);
	printf("numOverlap = %d\n", numOverlap);
	for (int i=0;i<numOverlap;i++)
	{
		printf("pair(%d,%d)\n",overlapCpu[i].x,overlapCpu[i].y);
	}
	delete bp;
	clReleaseCommandQueue(queue);
	clReleaseDevice(dev);
	clReleaseContext(ctx);
	
}
