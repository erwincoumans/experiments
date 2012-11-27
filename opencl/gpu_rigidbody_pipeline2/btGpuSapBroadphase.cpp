
#include "btGpuSapBroadphase.h"
#include "LinearMath/btVector3.h"
#include "../broadphase_benchmark/btLauncherCL.h"
#include "LinearMath/btQuickprof.h"
#include "../basic_initialize/btOpenCLUtils.h"
#define MSTRINGIFY(A) #A
static char* interopKernelString = 
#include "../broadphase_benchmark/integrateKernel.cl"
#include "../broadphase_benchmark/sapKernels.h"
#include "../broadphase_benchmark/sapFastKernels.h"



btGpuSapBroadphase::btGpuSapBroadphase(cl_context ctx,cl_device_id device, cl_command_queue  q )
:m_context(ctx),
m_device(device),
m_queue(q),
m_allAabbsGPU(ctx,q),
m_dynamicAabbsGPU(ctx,q),
m_staticAabbsGPU(ctx,q),
m_overlappingPairs(ctx,q),
m_gpuDynamicSortData(ctx,q),
m_gpuStaticSortData(ctx,q),
m_gpuDynamicSortedAabbs(ctx,q),
m_gpuStaticSortedAabbs(ctx,q)

{
	const char* sapSrc = sapCL;
    const char* sapFastSrc = sapFastCL;
    
	cl_int errNum=0;

	cl_program sapProg = btOpenCLUtils::compileCLProgramFromString(m_context,m_device,sapSrc,&errNum,"","../../opencl/broadphase_benchmark/sap.cl");
	btAssert(errNum==CL_SUCCESS);
	cl_program sapFastProg = btOpenCLUtils::compileCLProgramFromString(m_context,m_device,sapFastSrc,&errNum,"","../../opencl/broadphase_benchmark/sapFast.cl");
	btAssert(errNum==CL_SUCCESS);

	
	//m_sapKernel = btOpenCLUtils::compileCLKernelFromString(m_context, m_device,sapSrc, "computePairsKernelOriginal",&errNum,sapProg );
	//m_sapKernel = btOpenCLUtils::compileCLKernelFromString(m_context, m_device,sapSrc, "computePairsKernelBarrier",&errNum,sapProg );
	//m_sapKernel = btOpenCLUtils::compileCLKernelFromString(m_context, m_device,sapSrc, "computePairsKernelLocalSharedMemory",&errNum,sapProg );

	
	m_sap2Kernel = btOpenCLUtils::compileCLKernelFromString(m_context, m_device,sapSrc, "computePairsKernelTwoArrays",&errNum,sapProg );
	btAssert(errNum==CL_SUCCESS);

#if 0

	m_sapKernel = btOpenCLUtils::compileCLKernelFromString(m_context, m_device,sapSrc, "computePairsKernelOriginal",&errNum,sapProg );
	btAssert(errNum==CL_SUCCESS);
#else
#ifndef __APPLE__
	m_sapKernel = btOpenCLUtils::compileCLKernelFromString(m_context, m_device,sapFastSrc, "computePairsKernel",&errNum,sapFastProg );
	btAssert(errNum==CL_SUCCESS);
#else
	m_sapKernel = btOpenCLUtils::compileCLKernelFromString(m_context, m_device,sapSrc, "computePairsKernelLocalSharedMemory",&errNum,sapProg );
	btAssert(errNum==CL_SUCCESS);
#endif
#endif


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
	clReleaseKernel(m_sap2Kernel);

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

	btAssert(m_allAabbsCPU.size() == m_allAabbsGPU.size());
	

//#define FORCE_HOST
#ifdef FORCE_HOST
	

	btAlignedObjectArray<btSapAabb> allHostAabbs;
	m_allAabbsGPU.copyToHost(allHostAabbs);
	
	{
		int numDynamicAabbs = m_dynamicAabbsCPU.size();
		for (int j=0;j<numDynamicAabbs;j++)
		{
			//sync aabb
			int aabbIndex = m_dynamicAabbsCPU[j].m_signedMaxIndices[3];
			m_dynamicAabbsCPU[j] = allHostAabbs[aabbIndex];
			m_dynamicAabbsCPU[j].m_signedMaxIndices[3] = aabbIndex;
		}
	}

	{
		int numStaticAabbs = m_staticAabbsCPU.size();
		for (int j=0;j<numStaticAabbs;j++)
		{
			//sync aabb
			int aabbIndex = m_staticAabbsCPU[j].m_signedMaxIndices[3];
			m_staticAabbsCPU[j] = allHostAabbs[aabbIndex];
			m_staticAabbsCPU[j].m_signedMaxIndices[3] = aabbIndex;

		}
	}

	btAlignedObjectArray<btInt2> hostPairs;

	{
		int numDynamicAabbs = m_dynamicAabbsCPU.size();
		for (int i=0;i<numDynamicAabbs;i++)
		{
			float reference = m_dynamicAabbsCPU[i].m_max[axis];

			for (int j=i+1;j<numDynamicAabbs;j++)
			{
				if (TestAabbAgainstAabb2((btVector3&)m_dynamicAabbsCPU[i].m_min, (btVector3&)m_dynamicAabbsCPU[i].m_max,
					(btVector3&)m_dynamicAabbsCPU[j].m_min,(btVector3&)m_dynamicAabbsCPU[j].m_max))
				{
					btInt2 pair;
					pair.x = m_dynamicAabbsCPU[i].m_minIndices[3];//store the original index in the unsorted aabb array
					pair.y = m_dynamicAabbsCPU[j].m_minIndices[3];
					hostPairs.push_back(pair);
				}
			}
		}
	}

	
	{
		int numDynamicAabbs = m_dynamicAabbsCPU.size();
		for (int i=0;i<numDynamicAabbs;i++)
		{
			float reference = m_dynamicAabbsCPU[i].m_max[axis];
			int numStaticAabbs = m_staticAabbsCPU.size();

			for (int j=0;j<numStaticAabbs;j++)
			{
				if (TestAabbAgainstAabb2((btVector3&)m_dynamicAabbsCPU[i].m_min, (btVector3&)m_dynamicAabbsCPU[i].m_max,
					(btVector3&)m_staticAabbsCPU[j].m_min,(btVector3&)m_staticAabbsCPU[j].m_max))
				{
					btInt2 pair;
					pair.x = m_staticAabbsCPU[j].m_minIndices[3];
					pair.y = m_dynamicAabbsCPU[i].m_minIndices[3];//store the original index in the unsorted aabb array
					hostPairs.push_back(pair);
				}
			}
		}
	}


	if (hostPairs.size())
	{
		m_overlappingPairs.copyFromHost(hostPairs);
	} else
	{
		m_overlappingPairs.resize(0);
	}
#else
	{


	{
		BT_PROFILE("Synchronize m_dynamicAabbsGPU (CPU/slow)");
		btAlignedObjectArray<btSapAabb> allHostAabbs;
		m_allAabbsGPU.copyToHost(allHostAabbs);

		m_dynamicAabbsGPU.copyToHost(m_dynamicAabbsCPU);
		{
			int numDynamicAabbs = m_dynamicAabbsCPU.size();
			for (int j=0;j<numDynamicAabbs;j++)
			{
				//sync aabb
				int aabbIndex = m_dynamicAabbsCPU[j].m_signedMaxIndices[3];
				m_dynamicAabbsCPU[j] = allHostAabbs[aabbIndex];
				m_dynamicAabbsCPU[j].m_signedMaxIndices[3] = aabbIndex;
			}
		}
		m_dynamicAabbsGPU.copyFromHost(m_dynamicAabbsCPU);
	
	}




		BT_PROFILE("GPU SAP");
		
		int numDynamicAabbs = m_dynamicAabbsGPU.size();
		m_gpuDynamicSortData.resize(numDynamicAabbs);
		int numStaticAabbs = m_dynamicAabbsGPU.size();

#if 1
		if (m_dynamicAabbsGPU.size())
		{
			BT_PROFILE("flipFloatKernel");
			btBufferInfoCL bInfo[] = { btBufferInfoCL( m_dynamicAabbsGPU.getBufferCL(), true ), btBufferInfoCL( m_gpuDynamicSortData.getBufferCL())};
			btLauncherCL launcher(m_queue, m_flipFloatKernel );
			launcher.setBuffers( bInfo, sizeof(bInfo)/sizeof(btBufferInfoCL) );
			launcher.setConst( numDynamicAabbs  );
			launcher.setConst( axis  );
			
			int num = numDynamicAabbs;
			launcher.launch1D( num);
			clFinish(m_queue);
		}

		{
			BT_PROFILE("gpu radix sort\n");
			m_sorter->execute(m_gpuDynamicSortData);
			clFinish(m_queue);
		}

		m_gpuDynamicSortedAabbs.resize(numDynamicAabbs);
		if (numDynamicAabbs)
		{
			BT_PROFILE("scatterKernel");
			btBufferInfoCL bInfo[] = { btBufferInfoCL( m_dynamicAabbsGPU.getBufferCL(), true ), btBufferInfoCL( m_gpuDynamicSortData.getBufferCL(),true),btBufferInfoCL(m_gpuDynamicSortedAabbs.getBufferCL())};
			btLauncherCL launcher(m_queue, m_scatterKernel );
			launcher.setBuffers( bInfo, sizeof(bInfo)/sizeof(btBufferInfoCL) );
			launcher.setConst( numDynamicAabbs);
			int num = numDynamicAabbs;
			launcher.launch1D( num);
			clFinish(m_queue);
			
		}
        

			int maxPairsPerBody = 64;
			int maxPairs = maxPairsPerBody * numDynamicAabbs;//todo
			m_overlappingPairs.resize(maxPairs);

			btOpenCLArray<int> pairCount(m_context, m_queue);
			pairCount.push_back(0);
            int numPairs=0;

			{
				int numStaticAabbs = m_staticAabbsGPU.size();
				if (numStaticAabbs && numDynamicAabbs)
				{
					BT_PROFILE("sap2Kernel");
					btBufferInfoCL bInfo[] = { btBufferInfoCL( m_gpuDynamicSortedAabbs.getBufferCL() ),btBufferInfoCL( m_staticAabbsGPU.getBufferCL() ), btBufferInfoCL( m_overlappingPairs.getBufferCL() ), btBufferInfoCL(pairCount.getBufferCL())};
					btLauncherCL launcher(m_queue, m_sap2Kernel);
					launcher.setBuffers( bInfo, sizeof(bInfo)/sizeof(btBufferInfoCL) );
					launcher.setConst( numDynamicAabbs  );
					launcher.setConst( numStaticAabbs  );
					launcher.setConst( axis  );
					launcher.setConst( maxPairs  );

					int num = numDynamicAabbs;
					launcher.launch1D( num);
					clFinish(m_queue);
                
					numPairs = pairCount.at(0);
				}
			}
			if (m_gpuDynamicSortedAabbs.size())
			{
				BT_PROFILE("sapKernel");
				btBufferInfoCL bInfo[] = { btBufferInfoCL( m_gpuDynamicSortedAabbs.getBufferCL() ), btBufferInfoCL( m_overlappingPairs.getBufferCL() ), btBufferInfoCL(pairCount.getBufferCL())};
				btLauncherCL launcher(m_queue, m_sapKernel);
				launcher.setBuffers( bInfo, sizeof(bInfo)/sizeof(btBufferInfoCL) );
				launcher.setConst( numDynamicAabbs  );
				launcher.setConst( axis  );
				launcher.setConst( maxPairs  );

			
				int num = numDynamicAabbs;
#if 0                
                int buffSize = launcher.getSerializationBufferSize();
                unsigned char* buf = new unsigned char[buffSize+sizeof(int)];
                for (int i=0;i<buffSize+1;i++)
                {
                    unsigned char* ptr = (unsigned char*)&buf[i];
                    *ptr = 0xff;
                }
                int actualWrite = launcher.serializeArguments(buf,buffSize);
                
                unsigned char* cptr = (unsigned char*)&buf[buffSize];
    //            printf("buf[buffSize] = %d\n",*cptr);
                
                assert(buf[buffSize]==0xff);//check for buffer overrun
                int* ptr = (int*)&buf[buffSize];
                
                *ptr = num;
                
                FILE* f = fopen("m_sapKernelArgs.bin","wb");
                fwrite(buf,buffSize+sizeof(int),1,f);
                fclose(f);
#endif//

                launcher.launch1D( num);
				clFinish(m_queue);
                
                numPairs = pairCount.at(0);
                
			}
			
#else
        int numPairs = 0;
        
        
        btLauncherCL launcher(m_queue, m_sapKernel);

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
            int serializedBytes = launcher.deserializeArgs(buf, sizeInBytes,m_context);
            int num = *(int*)&buf[serializedBytes];
            launcher.launch1D( num);
            
            btOpenCLArray<int> pairCount(m_context, m_queue);
            int numElements = launcher.m_arrays[2]->size()/sizeof(int);
            pairCount.setFromOpenCLBuffer(launcher.m_arrays[2]->getBufferCL(),numElements);
            numPairs = pairCount.at(0);
            //printf("overlapping pairs = %d\n",numPairs);
            btAlignedObjectArray<btInt2>		hostOoverlappingPairs;
            btOpenCLArray<btInt2> tmpGpuPairs(m_context,m_queue);
            tmpGpuPairs.setFromOpenCLBuffer(launcher.m_arrays[1]->getBufferCL(),numPairs );
   
            tmpGpuPairs.copyToHost(hostOoverlappingPairs);
            m_overlappingPairs.copyFromHost(hostOoverlappingPairs);
            //printf("hello %d\n", m_overlappingPairs.size());
            free(buf);
            fclose(f);
            
        } else {
            printf("error: cannot find file %s\n",fileName);
        }
        
        clFinish(m_queue);

        
#endif

			
        m_overlappingPairs.resize(numPairs);
		
	}//BT_PROFILE("GPU_RADIX SORT");

#endif
}

void btGpuSapBroadphase::writeAabbsToGpu()
{
	m_allAabbsGPU.copyFromHost(m_allAabbsCPU);//might not be necessary, the 'setupGpuAabbsFull' already takes care of this
	m_dynamicAabbsGPU.copyFromHost(m_dynamicAabbsCPU);
	m_staticAabbsGPU.copyFromHost(m_staticAabbsCPU);

}

void btGpuSapBroadphase::createStaticProxy(const btVector3& aabbMin,  const btVector3& aabbMax, int userPtr ,short int collisionFilterGroup,short int collisionFilterMask)
{
	int index = userPtr;
	btSapAabb aabb;
	for (int i=0;i<4;i++)
	{
		aabb.m_min[i] = aabbMin[i];
		aabb.m_max[i] = aabbMax[i];
	}
	aabb.m_minIndices[3] = index;
	aabb.m_signedMaxIndices[3] = m_allAabbsCPU.size();
	m_staticAabbsCPU.push_back(aabb);
	m_allAabbsCPU.push_back(aabb);
}

void btGpuSapBroadphase::createProxy(const btVector3& aabbMin,  const btVector3& aabbMax, int userPtr ,short int collisionFilterGroup,short int collisionFilterMask)
{
	int index = userPtr;
	btSapAabb aabb;
	for (int i=0;i<4;i++)
	{
		aabb.m_min[i] = aabbMin[i];
		aabb.m_max[i] = aabbMax[i];
	}
	aabb.m_minIndices[3] = index;
	aabb.m_signedMaxIndices[3] = m_allAabbsCPU.size();
	m_dynamicAabbsCPU.push_back(aabb);
	m_allAabbsCPU.push_back(aabb);
}

cl_mem	btGpuSapBroadphase::getAabbBuffer()
{
	return m_allAabbsGPU.getBufferCL();
}

int	btGpuSapBroadphase::getNumOverlap()
{
	return m_overlappingPairs.size();
}
cl_mem	btGpuSapBroadphase::getOverlappingPairBuffer()
{
	return m_overlappingPairs.getBufferCL();
}
