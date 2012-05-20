
#include "btGpuSapBroadphase.h"
#include "LinearMath/btVector3.h"
#include "../broadphase_benchmark/btLauncherCL.h"
#include "LinearMath/btQuickprof.h"
#include "../basic_initialize/btOpenCLUtils.h"
#define MSTRINGIFY(A) #A
static char* interopKernelString = 
#include "../broadphase_benchmark/integrateKernel.cl"



btGpuSapBroadphase::btGpuSapBroadphase(cl_context ctx,cl_device_id device, cl_command_queue  q )
:m_context(ctx),
m_device(device),
m_queue(q),
m_aabbs(ctx,q),
m_overlappingPairs(ctx,q),
m_gpuSortData(ctx,q),
m_gpuSortedAabbs(ctx,q)
{
	char* src = 0;
	cl_int errNum=0;

	cl_program sapProg = btOpenCLUtils::compileCLProgramFromString(m_context,m_device,src,&errNum,"","../../opencl/broadphase_benchmark/sap.cl");
	btAssert(errNum==CL_SUCCESS);

	
	
	//m_sapKernel = btOpenCLUtils::compileCLKernelFromString(m_context, m_device,interopKernelString, "computePairsKernelOriginal",&errNum,sapProg );
	//m_sapKernel = btOpenCLUtils::compileCLKernelFromString(m_context, m_device,interopKernelString, "computePairsKernelBarrier",&errNum,sapProg );
	//m_sapKernel = btOpenCLUtils::compileCLKernelFromString(m_context, m_device,interopKernelString, "computePairsKernelLocalSharedMemory",&errNum,sapProg );
	m_sapKernel = btOpenCLUtils::compileCLKernelFromString(m_context, m_device,interopKernelString, "computePairsKernel",&errNum,sapProg );
	btAssert(errNum==CL_SUCCESS);

	m_flipFloatKernel = btOpenCLUtils::compileCLKernelFromString(m_context, m_device,interopKernelString, "flipFloatKernel",&errNum,sapProg );

	m_scatterKernel = btOpenCLUtils::compileCLKernelFromString(m_context, m_device,interopKernelString, "scatterKernel",&errNum,sapProg );

	m_sorter = new btRadixSort32CL(m_context,m_device,m_queue);
}

btGpuSapBroadphase::~btGpuSapBroadphase()
{
	delete m_sorter;
	clReleaseKernel(m_scatterKernel);
	clReleaseKernel(m_flipFloatKernel);
	clReleaseKernel(m_sapKernel);

}

/// conservative test for overlap between two aabbs
static bool TestAabbAgainstAabb2(const btVector3 &aabbMin1, const btVector3 &aabbMax1,
								const btVector3 &aabbMin2, const btVector3 &aabbMax2)
{
	bool overlap = true;
	overlap = (aabbMin1.getX() > aabbMax2.getX() || aabbMax1.getX() < aabbMin2.getX()) ? false : overlap;
	overlap = (aabbMin1.getZ() > aabbMax2.getZ() || aabbMax1.getZ() < aabbMin2.getZ()) ? false : overlap;
	overlap = (aabbMin1.getY() > aabbMax2.getY() || aabbMax1.getY() < aabbMin2.getY()) ? false : overlap;
	return overlap;
}

void  btGpuSapBroadphase::calculateOverlappingPairs()
{
	int axis = 0;//todo on GPU for now hardcode

//#define FORCE_HOST
#ifdef FORCE_HOST
	

	btAlignedObjectArray<btSapAabb> hostAabbs;
	m_aabbs.copyToHost(hostAabbs);
	int numAabbs = hostAabbs.size();

	btAlignedObjectArray<btInt2> hostPairs;
	for (int i=0;i<hostAabbs.size();i++)
	{
		float reference = hostAabbs[i].m_max[axis];

		for (int j=i+1;j<numAabbs;j++)
		{
			if (TestAabbAgainstAabb2((btVector3&)hostAabbs[i].m_min, (btVector3&)hostAabbs[i].m_max,
				(btVector3&)hostAabbs[j].m_min,(btVector3&)hostAabbs[j].m_max))
			{
				btInt2 pair;
				pair.x = hostAabbs[i].m_minIndices[3];//store the original index in the unsorted aabb array
				pair.y = hostAabbs[j].m_minIndices[3];
				hostPairs.push_back(pair);
			}
		}
	}

	if (hostPairs.size())
	{
		m_overlappingPairs.copyFromHost(hostPairs);
	}
#else
	{
		BT_PROFILE("GPU SAP");
		
		int numAabbs = m_aabbs.size();
		m_gpuSortData.resize(numAabbs);
	
		{
			BT_PROFILE("flipFloatKernel");
			btBufferInfoCL bInfo[] = { btBufferInfoCL( m_aabbs.getBufferCL(), true ), btBufferInfoCL( m_gpuSortData.getBufferCL())};
			btLauncherCL launcher(m_queue, m_flipFloatKernel );
			launcher.setBuffers( bInfo, sizeof(bInfo)/sizeof(btBufferInfoCL) );
			launcher.setConst( numAabbs  );
			launcher.setConst( axis  );
			
			int num = numAabbs;
			launcher.launch1D( num);
			clFinish(m_queue);
		}

		{
			BT_PROFILE("gpu radix sort\n");
			m_sorter->execute(m_gpuSortData);
			clFinish(m_queue);
		}

		m_gpuSortedAabbs.resize(numAabbs);
		{
			BT_PROFILE("scatterKernel");
			btBufferInfoCL bInfo[] = { btBufferInfoCL( m_aabbs.getBufferCL(), true ), btBufferInfoCL( m_gpuSortData.getBufferCL(),true),btBufferInfoCL(m_gpuSortedAabbs.getBufferCL())};
			btLauncherCL launcher(m_queue, m_scatterKernel );
			launcher.setBuffers( bInfo, sizeof(bInfo)/sizeof(btBufferInfoCL) );
			launcher.setConst( numAabbs);
			int num = numAabbs;
			launcher.launch1D( num);
			clFinish(m_queue);
			
		}


			int maxPairsPerBody = 64;
			int maxPairs = maxPairsPerBody * numAabbs;//todo
			m_overlappingPairs.resize(maxPairs);

			btOpenCLArray<int> pairCount(m_context, m_queue);
			pairCount.push_back(0);

			{
				BT_PROFILE("sapKernel");
				btBufferInfoCL bInfo[] = { btBufferInfoCL( m_gpuSortedAabbs.getBufferCL(), true ), btBufferInfoCL( m_overlappingPairs.getBufferCL() ), btBufferInfoCL(pairCount.getBufferCL())};
				btLauncherCL launcher(m_queue, m_sapKernel);
				launcher.setBuffers( bInfo, sizeof(bInfo)/sizeof(btBufferInfoCL) );
				launcher.setConst( numAabbs  );
				launcher.setConst( axis  );
				launcher.setConst( maxPairs  );

			
				int num = numAabbs;
				launcher.launch1D( num);
				clFinish(m_queue);
			}
			
			int numPairs = pairCount.at(0);
			m_overlappingPairs.resize(numPairs);
		
	}//BT_PROFILE("GPU_RADIX SORT");

#endif
}

void btGpuSapBroadphase::createProxy(const btVector3& aabbMin,  const btVector3& aabbMax,int shapeType,
				void* userPtr ,short int collisionFilterGroup,short int collisionFilterMask)
{
	int index = (int)userPtr;
	btSapAabb aabb;
	for (int i=0;i<4;i++)
	{
		aabb.m_min[i] = aabbMin[i];
		aabb.m_max[i] = aabbMax[i];
	}
	aabb.m_minIndices[3] = index;//m_aabbs.size();
	m_aabbs.push_back(aabb);
}

cl_mem	btGpuSapBroadphase::getAabbBuffer()
{
	return m_aabbs.getBufferCL();
}

int	btGpuSapBroadphase::getNumOverlap()
{
	return m_overlappingPairs.size();
}
cl_mem	btGpuSapBroadphase::getOverlappingPairBuffer()
{
	return m_overlappingPairs.getBufferCL();
}

