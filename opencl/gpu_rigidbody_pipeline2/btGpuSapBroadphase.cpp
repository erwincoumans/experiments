
#include "btGpuSapBroadphase.h"
#include "LinearMath/btVector3.h"
#include "../broadphase_benchmark/btLauncherCL.h"
#include "LinearMath/btQuickprof.h"
#include "../basic_initialize/btOpenCLUtils.h"
#define MSTRINGIFY(A) #A
static char* interopKernelString = 
#include "../broadphase_benchmark/integrateKernel.cl"
#include "../broadphase_benchmark/sapKernels.h"



btGpuSapBroadphase::btGpuSapBroadphase(cl_context ctx,cl_device_id device, cl_command_queue  q )
:m_context(ctx),
m_device(device),
m_queue(q),
m_aabbsGPU(ctx,q),
m_overlappingPairs(ctx,q),
m_gpuSortData(ctx,q),
m_gpuSortedAabbs(ctx,q)
{
	const char* sapSrc = sapCL;
	cl_int errNum=0;

	cl_program sapProg = btOpenCLUtils::compileCLProgramFromString(m_context,m_device,sapSrc,&errNum,"","../../opencl/broadphase_benchmark/sap.cl");
	btAssert(errNum==CL_SUCCESS);

	
	
	//m_sapKernel = btOpenCLUtils::compileCLKernelFromString(m_context, m_device,sapSrc, "computePairsKernelOriginal",&errNum,sapProg );
	//m_sapKernel = btOpenCLUtils::compileCLKernelFromString(m_context, m_device,sapSrc, "computePairsKernelBarrier",&errNum,sapProg );
	//m_sapKernel = btOpenCLUtils::compileCLKernelFromString(m_context, m_device,sapSrc, "computePairsKernelLocalSharedMemory",&errNum,sapProg );
	m_sapKernel = btOpenCLUtils::compileCLKernelFromString(m_context, m_device,sapSrc, "computePairsKernel",&errNum,sapProg );
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

	btAssert(m_aabbsCPU.size() == m_aabbsGPU.size());
	

//#define FORCE_HOST
#ifdef FORCE_HOST
	

	btAlignedObjectArray<btSapAabb> hostAabbs;
	m_aabbsGPU.copyToHost(hostAabbs);
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
		
		int numAabbs = m_aabbsGPU.size();
		m_gpuSortData.resize(numAabbs);
#if 1
		{
			BT_PROFILE("flipFloatKernel");
			btBufferInfoCL bInfo[] = { btBufferInfoCL( m_aabbsGPU.getBufferCL(), true ), btBufferInfoCL( m_gpuSortData.getBufferCL())};
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
			btBufferInfoCL bInfo[] = { btBufferInfoCL( m_aabbsGPU.getBufferCL(), true ), btBufferInfoCL( m_gpuSortData.getBufferCL(),true),btBufferInfoCL(m_gpuSortedAabbs.getBufferCL())};
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
            int numPairs=0;

			{
				BT_PROFILE("sapKernel");
				btBufferInfoCL bInfo[] = { btBufferInfoCL( m_gpuSortedAabbs.getBufferCL() ), btBufferInfoCL( m_overlappingPairs.getBufferCL() ), btBufferInfoCL(pairCount.getBufferCL())};
				btLauncherCL launcher(m_queue, m_sapKernel);
				launcher.setBuffers( bInfo, sizeof(bInfo)/sizeof(btBufferInfoCL) );
				launcher.setConst( numAabbs  );
				launcher.setConst( axis  );
				launcher.setConst( maxPairs  );

			
				int num = numAabbs;
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
	m_aabbsGPU.copyFromHost(m_aabbsCPU);
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
	m_aabbsCPU.push_back(aabb);
}

cl_mem	btGpuSapBroadphase::getAabbBuffer()
{
	return m_aabbsGPU.getBufferCL();
}

int	btGpuSapBroadphase::getNumOverlap()
{
	return m_overlappingPairs.size();
}
cl_mem	btGpuSapBroadphase::getOverlappingPairBuffer()
{
	return m_overlappingPairs.getBufferCL();
}

