/*
Bullet Continuous Collision Detection and Physics Library, http://bulletphysics.org
Copyright (C) 2006 - 2009 Sony Computer Entertainment Inc. 

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/


#include "LinearMath/btAlignedAllocator.h"
#include "LinearMath/btQuickprof.h"
#include "BulletCollision/BroadphaseCollision/btOverlappingPairCache.h"

#include "bt3dGridBroadphaseOCL.h"

#include <stdio.h>
#include <string.h>
#include "Adl/Adl.h"
#include <AdlPrimitives/Scan/PrefixScan.h>




#define MSTRINGIFY(A) #A

static const char* spProgramSource = 
#include "bt3dGridBroadphaseOCL.cl"

adl::PrefixScan<adl::TYPE_CL>::Data* gData1=0;
adl::Buffer<unsigned int>* m_srcClBuffer=0;

bt3dGridBroadphaseOCL::bt3dGridBroadphaseOCL(	btOverlappingPairCache* overlappingPairCache,
												const btVector3& cellSize, 
												int gridSizeX, int gridSizeY, int gridSizeZ, 
												int maxSmallProxies, int maxLargeProxies, int maxPairsPerSmallProxy,
												btScalar maxSmallProxySize,
												int maxSmallProxiesPerCell,
												cl_context context, cl_device_id device, cl_command_queue queue) : 
	btGpu3DGridBroadphase(overlappingPairCache, cellSize, gridSizeX, gridSizeY, gridSizeZ, maxSmallProxies, maxLargeProxies, maxPairsPerSmallProxy, maxSmallProxySize, maxSmallProxiesPerCell)
{
	initCL(context, device, queue);
	allocateBuffers();
	prefillBuffers();
	initKernels();

	//create an Adl device host and OpenCL device

	adl::DeviceUtils::Config cfg;
	m_deviceHost = adl::DeviceUtils::allocate( adl::TYPE_HOST, cfg );

	m_deviceCL = new adl::DeviceCL();
	m_deviceCL->m_deviceIdx = device;
	m_deviceCL->m_context = context;
	m_deviceCL->m_commandQueue = queue;
	m_deviceCL->m_kernelManager = new adl::KernelManager;

	m_srcClBuffer = new adl::Buffer<unsigned int> (m_deviceCL,maxSmallProxies+2);
	gData1 = adl::PrefixScan<adl::TYPE_CL>::allocate( m_deviceCL, maxSmallProxies+2,adl::PrefixScanBase::EXCLUSIVE );

}



bt3dGridBroadphaseOCL::~bt3dGridBroadphaseOCL()
{
	//btSimpleBroadphase will free memory of btSortedOverlappingPairCache, because m_ownsPairCache
	assert(m_bInitialized);
	adl::DeviceUtils::deallocate(m_deviceHost);
	adl::PrefixScan<adl::TYPE_CL>::deallocate(gData1);
	delete m_deviceCL;
	

}

#ifdef CL_PLATFORM_MINI_CL
// there is a problem with MSVC9 : static constructors are not called if variables defined in library and are not used
// looks like it is because of optimization
// probably this will happen with other compilers as well
// so to make it robust, register kernels again (it is safe)
#define MINICL_DECLARE(a) extern "C" void a();
MINICL_DECLARE(kCalcHashAABB)
MINICL_DECLARE(kClearCellStart)
MINICL_DECLARE(kFindCellStart)
MINICL_DECLARE(kFindOverlappingPairs)
MINICL_DECLARE(kFindPairsLarge)
MINICL_DECLARE(kComputePairCacheChanges)
MINICL_DECLARE(kSqueezeOverlappingPairBuff)
MINICL_DECLARE(kBitonicSortCellIdLocal)
MINICL_DECLARE(kBitonicSortCellIdLocal1)
MINICL_DECLARE(kBitonicSortCellIdMergeGlobal)
MINICL_DECLARE(kBitonicSortCellIdMergeLocal)
#undef MINICL_DECLARE
#endif

void bt3dGridBroadphaseOCL::initCL(cl_context context, cl_device_id device, cl_command_queue queue)
{

	#ifdef CL_PLATFORM_MINI_CL
		// call constructors here
		MINICL_REGISTER(kCalcHashAABB)
		MINICL_REGISTER(kClearCellStart)
		MINICL_REGISTER(kFindCellStart)
		MINICL_REGISTER(kFindOverlappingPairs)
		MINICL_REGISTER(kFindPairsLarge)
		MINICL_REGISTER(kComputePairCacheChanges)
		MINICL_REGISTER(kSqueezeOverlappingPairBuff)
		MINICL_REGISTER(kBitonicSortCellIdLocal)
		MINICL_REGISTER(kBitonicSortCellIdLocal1)
		MINICL_REGISTER(kBitonicSortCellIdMergeGlobal)
		MINICL_REGISTER(kBitonicSortCellIdMergeLocal)
	#endif

	cl_int ciErrNum;

	btAssert(context);
	m_cxMainContext = context;
	btAssert(device);
	m_cdDevice = device;
	btAssert(queue);
	m_cqCommandQue = queue;
	
	// create the program
	size_t programLength = strlen(spProgramSource);
	printf("OpenCL compiles bt3dGridBroadphaseOCL.cl ...");
	m_cpProgram = clCreateProgramWithSource(m_cxMainContext, 1, &spProgramSource, &programLength, &ciErrNum);
	// build the program
	ciErrNum = clBuildProgram(m_cpProgram, 0, NULL, "-DGUID_ARG=""""", NULL, NULL);
	if(ciErrNum != CL_SUCCESS)
	{
		// write out standard error
		char cBuildLog[10240];
		clGetProgramBuildInfo(m_cpProgram, m_cdDevice, CL_PROGRAM_BUILD_LOG, 
							  sizeof(cBuildLog), cBuildLog, NULL );
		printf("\n\n%s\n\n\n", cBuildLog);
		printf("Press ENTER key to terminate the program\n");
		getchar();
		exit(-1); 
	}
	printf("OK\n");
}


void bt3dGridBroadphaseOCL::initKernels()
{
	initKernel(GRID3DOCL_KERNEL_CALC_HASH_AABB,	"kCalcHashAABB");
	setKernelArg(GRID3DOCL_KERNEL_CALC_HASH_AABB, 1, sizeof(cl_mem),(void*)&m_dAABB);
	setKernelArg(GRID3DOCL_KERNEL_CALC_HASH_AABB, 2, sizeof(cl_mem),(void*)&m_dBodiesHash);
	setKernelArg(GRID3DOCL_KERNEL_CALC_HASH_AABB, 3, sizeof(cl_mem),(void*)&m_dBpParams);

	initKernel(GRID3DOCL_KERNEL_CLEAR_CELL_START, "kClearCellStart");
	setKernelArg(GRID3DOCL_KERNEL_CLEAR_CELL_START, 1, sizeof(cl_mem),(void*)&m_dCellStart);

	initKernel(GRID3DOCL_KERNEL_FIND_CELL_START, "kFindCellStart");
	setKernelArg(GRID3DOCL_KERNEL_FIND_CELL_START, 1, sizeof(cl_mem),(void*)&m_dBodiesHash);
	setKernelArg(GRID3DOCL_KERNEL_FIND_CELL_START, 2, sizeof(cl_mem),(void*)&m_dCellStart);

	initKernel(GRID3DOCL_KERNEL_FIND_OVERLAPPING_PAIRS, "kFindOverlappingPairs");
	setKernelArg(GRID3DOCL_KERNEL_FIND_OVERLAPPING_PAIRS, 1, sizeof(cl_mem),(void*)&m_dAABB);
	setKernelArg(GRID3DOCL_KERNEL_FIND_OVERLAPPING_PAIRS, 2, sizeof(cl_mem),(void*)&m_dBodiesHash);
	setKernelArg(GRID3DOCL_KERNEL_FIND_OVERLAPPING_PAIRS, 3, sizeof(cl_mem),(void*)&m_dCellStart);
	setKernelArg(GRID3DOCL_KERNEL_FIND_OVERLAPPING_PAIRS, 4, sizeof(cl_mem),(void*)&m_dPairBuff);
	setKernelArg(GRID3DOCL_KERNEL_FIND_OVERLAPPING_PAIRS, 5, sizeof(cl_mem),(void*)&m_dPairBuffStartCurr);
	setKernelArg(GRID3DOCL_KERNEL_FIND_OVERLAPPING_PAIRS, 6, sizeof(cl_mem),(void*)&m_dBpParams);

	initKernel(GRID3DOCL_KERNEL_FIND_PAIRS_LARGE, "kFindPairsLarge");
	setKernelArg(GRID3DOCL_KERNEL_FIND_PAIRS_LARGE, 1, sizeof(cl_mem),(void*)&m_dAABB);
	setKernelArg(GRID3DOCL_KERNEL_FIND_PAIRS_LARGE, 2, sizeof(cl_mem),(void*)&m_dBodiesHash);
	setKernelArg(GRID3DOCL_KERNEL_FIND_PAIRS_LARGE, 3, sizeof(cl_mem),(void*)&m_dCellStart);
	setKernelArg(GRID3DOCL_KERNEL_FIND_PAIRS_LARGE, 4, sizeof(cl_mem),(void*)&m_dPairBuff);
	setKernelArg(GRID3DOCL_KERNEL_FIND_PAIRS_LARGE, 5, sizeof(cl_mem),(void*)&m_dPairBuffStartCurr);

	initKernel(GRID3DOCL_KERNEL_COMPUTE_CACHE_CHANGES, "kComputePairCacheChanges");
	setKernelArg(GRID3DOCL_KERNEL_COMPUTE_CACHE_CHANGES, 1, sizeof(cl_mem),(void*)&m_dPairBuff);
	setKernelArg(GRID3DOCL_KERNEL_COMPUTE_CACHE_CHANGES, 2, sizeof(cl_mem),(void*)&m_dPairBuffStartCurr);
	setKernelArg(GRID3DOCL_KERNEL_COMPUTE_CACHE_CHANGES, 3, sizeof(cl_mem),(void*)&m_dPairScan);
	setKernelArg(GRID3DOCL_KERNEL_COMPUTE_CACHE_CHANGES, 4, sizeof(cl_mem),(void*)&m_dAABB);

	initKernel(GRID3DOCL_KERNEL_SQUEEZE_PAIR_BUFF, "kSqueezeOverlappingPairBuff");
	setKernelArg(GRID3DOCL_KERNEL_SQUEEZE_PAIR_BUFF, 1, sizeof(cl_mem),(void*)&m_dPairBuff);
	setKernelArg(GRID3DOCL_KERNEL_SQUEEZE_PAIR_BUFF, 2, sizeof(cl_mem),(void*)&m_dPairBuffStartCurr);
	setKernelArg(GRID3DOCL_KERNEL_SQUEEZE_PAIR_BUFF, 3, sizeof(cl_mem),(void*)&m_dPairScan);
	setKernelArg(GRID3DOCL_KERNEL_SQUEEZE_PAIR_BUFF, 4, sizeof(cl_mem),(void*)&m_dPairOut);
	setKernelArg(GRID3DOCL_KERNEL_SQUEEZE_PAIR_BUFF, 5, sizeof(cl_mem),(void*)&m_dAABB);

	initKernel(GRID3DOCL_KERNEL_BITONIC_SORT_CELL_ID_LOCAL, "kBitonicSortCellIdLocal");
	initKernel(GRID3DOCL_KERNEL_BITONIC_SORT_CELL_ID_LOCAL_1, "kBitonicSortCellIdLocal1");
	initKernel(GRID3DOCL_KERNEL_BITONIC_SORT_CELL_ID_MERGE_GLOBAL, "kBitonicSortCellIdMergeGlobal");
	initKernel(GRID3DOCL_KERNEL_BITONIC_SORT_CELL_ID_MERGE_LOCAL, "kBitonicSortCellIdMergeLocal");
}


void bt3dGridBroadphaseOCL::allocateBuffers()
{
    cl_int ciErrNum;
    unsigned int memSize;
	// current version of bitonic sort works for power of 2 arrays only, so ...
	m_hashSize = 1;
	for(int bit = 1; bit < 32; bit++)
	{
		if(m_hashSize >= m_maxHandles)
		{
			break;
		}
		m_hashSize <<= 1;
	}
	memSize = m_hashSize * 2 * sizeof(unsigned int);
	m_dBodiesHash = clCreateBuffer(m_cxMainContext, CL_MEM_READ_WRITE, memSize, NULL, &ciErrNum);
	GRID3DOCL_CHECKERROR(ciErrNum, CL_SUCCESS);

	memSize = m_numCells * sizeof(unsigned int);
	m_dCellStart = clCreateBuffer(m_cxMainContext, CL_MEM_READ_WRITE, memSize, NULL, &ciErrNum);
	GRID3DOCL_CHECKERROR(ciErrNum, CL_SUCCESS);

	memSize = m_maxHandles * m_maxPairsPerBody * sizeof(unsigned int);
	m_dPairBuff = clCreateBuffer(m_cxMainContext, CL_MEM_READ_WRITE, memSize, NULL, &ciErrNum);
	GRID3DOCL_CHECKERROR(ciErrNum, CL_SUCCESS);

	memSize = (m_maxHandles * 2 + 1) * sizeof(unsigned int);
	m_dPairBuffStartCurr = clCreateBuffer(m_cxMainContext, CL_MEM_READ_WRITE, memSize, NULL, &ciErrNum);
	GRID3DOCL_CHECKERROR(ciErrNum, CL_SUCCESS);

	unsigned int numAABB = m_maxHandles + m_maxLargeHandles;
	memSize = numAABB * sizeof(float) * 4 * 2;
	m_dAABB = clCreateBuffer(m_cxMainContext, CL_MEM_READ_WRITE, memSize, NULL, &ciErrNum);
	GRID3DOCL_CHECKERROR(ciErrNum, CL_SUCCESS);

	memSize = (m_maxHandles + 2) * sizeof(unsigned int);
	m_dPairScan = clCreateBuffer(m_cxMainContext, CL_MEM_READ_WRITE, memSize, NULL, &ciErrNum);
	GRID3DOCL_CHECKERROR(ciErrNum, CL_SUCCESS);

	memSize = m_maxHandles * m_maxPairsPerBody * sizeof(unsigned int);
	m_dPairOut = clCreateBuffer(m_cxMainContext, CL_MEM_READ_WRITE, memSize, NULL, &ciErrNum);
	GRID3DOCL_CHECKERROR(ciErrNum, CL_SUCCESS);

	memSize = 3 * 4 * sizeof(float);
	m_dBpParams = clCreateBuffer(m_cxMainContext, CL_MEM_READ_WRITE, memSize, NULL, &ciErrNum);
	GRID3DOCL_CHECKERROR(ciErrNum, CL_SUCCESS);
}

void bt3dGridBroadphaseOCL::prefillBuffers()
{
	memset(m_hBodiesHash, 0x7F, m_maxHandles*2*sizeof(unsigned int));
	copyArrayToDevice(m_dBodiesHash, m_hBodiesHash, m_maxHandles * 2 * sizeof(unsigned int));
	// now fill the rest (bitonic sorting works with size == pow of 2)
	int remainder = m_hashSize - m_maxHandles;
	if(remainder)
	{
		copyArrayToDevice(m_dBodiesHash, m_hBodiesHash, remainder * 2 * sizeof(unsigned int), m_maxHandles * 2 * sizeof(unsigned int), 0);
	}
	copyArrayToDevice(m_dPairBuffStartCurr, m_hPairBuffStartCurr, (m_maxHandles * 2 + 1) * sizeof(unsigned int)); 
	memset(m_hPairBuff, 0x00, m_maxHandles * m_maxPairsPerBody * sizeof(unsigned int));
	copyArrayToDevice(m_dPairBuff, m_hPairBuff, m_maxHandles * m_maxPairsPerBody * sizeof(unsigned int));
}


void bt3dGridBroadphaseOCL::initKernel(int kernelId, char* pName)
{
	
	cl_int ciErrNum;
	cl_kernel kernel = clCreateKernel(m_cpProgram, pName, &ciErrNum);
	GRID3DOCL_CHECKERROR(ciErrNum, CL_SUCCESS);
	size_t wgSize;
	ciErrNum = clGetKernelWorkGroupInfo(kernel, m_cdDevice, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &wgSize, NULL);
	GRID3DOCL_CHECKERROR(ciErrNum, CL_SUCCESS);
	m_kernels[kernelId].m_Id = kernelId;
	m_kernels[kernelId].m_kernel = kernel;
	m_kernels[kernelId].m_name = pName;
	m_kernels[kernelId].m_workgroupSize = (int)wgSize;
	return;
}

void bt3dGridBroadphaseOCL::runKernelWithWorkgroupSize(int kernelId, int globalSize)
{
	if(globalSize <= 0)
	{
		return;
	}
	cl_kernel kernelFunc = m_kernels[kernelId].m_kernel;
	cl_int ciErrNum = clSetKernelArg(kernelFunc, 0, sizeof(int), (void*)&globalSize);
	GRID3DOCL_CHECKERROR(ciErrNum, CL_SUCCESS);
	int workgroupSize = m_kernels[kernelId].m_workgroupSize;
	if(workgroupSize <= 0)
	{ // let OpenCL library calculate workgroup size
		size_t globalWorkSize[2];
		globalWorkSize[0] = globalSize;
		globalWorkSize[1] = 1;
		ciErrNum = clEnqueueNDRangeKernel(m_cqCommandQue, kernelFunc, 1, NULL, globalWorkSize, NULL, 0,0,0 );
	}
	else
	{
		size_t localWorkSize[2], globalWorkSize[2];
		workgroupSize = btMin(workgroupSize, globalSize);
		int num_t = globalSize / workgroupSize;
		int num_g = num_t * workgroupSize;
		if(num_g < globalSize)
		{
			num_t++;
		}
		localWorkSize[0]  = workgroupSize;
		globalWorkSize[0] = num_t * workgroupSize;
		localWorkSize[1] = 1;
		globalWorkSize[1] = 1;
		ciErrNum = clEnqueueNDRangeKernel(m_cqCommandQue, kernelFunc, 1, NULL, globalWorkSize, localWorkSize, 0,0,0 );
	}
	GRID3DOCL_CHECKERROR(ciErrNum, CL_SUCCESS);
	ciErrNum = clFlush(m_cqCommandQue);
	GRID3DOCL_CHECKERROR(ciErrNum, CL_SUCCESS);
}


void bt3dGridBroadphaseOCL::setKernelArg(int kernelId, int argNum, int argSize, void* argPtr)
{
    cl_int ciErrNum;
	ciErrNum  = clSetKernelArg(m_kernels[kernelId].m_kernel, argNum, argSize, argPtr);
	GRID3DOCL_CHECKERROR(ciErrNum, CL_SUCCESS);
}


void bt3dGridBroadphaseOCL::copyArrayToDevice(cl_mem device, const void* host, unsigned int size, int devOffs, int hostOffs)
{
    cl_int ciErrNum;
	char* pHost = (char*)host + hostOffs;
    ciErrNum = clEnqueueWriteBuffer(m_cqCommandQue, device, CL_TRUE, devOffs, size, pHost, 0, NULL, NULL);
	GRID3DOCL_CHECKERROR(ciErrNum, CL_SUCCESS);
}

void bt3dGridBroadphaseOCL::copyArrayFromDevice(void* host, const cl_mem device, unsigned int size, int hostOffs, int devOffs)
{
    cl_int ciErrNum;
	char* pHost = (char*)host + hostOffs;
    ciErrNum = clEnqueueReadBuffer(m_cqCommandQue, device, CL_TRUE, devOffs, size, pHost, 0, NULL, NULL);
	GRID3DOCL_CHECKERROR(ciErrNum, CL_SUCCESS);
}



//
// overrides
//


void bt3dGridBroadphaseOCL::prepareAABB()
{
	btGpu3DGridBroadphase::prepareAABB();
	copyArrayToDevice(m_dAABB, m_hAABB, sizeof(bt3DGrid3F1U) * 2 * (m_numHandles + m_numLargeHandles)); 
	return;
}

void bt3dGridBroadphaseOCL::setParameters(bt3DGridBroadphaseParams* hostParams)
{
	btGpu3DGridBroadphase::setParameters(hostParams);
	struct btParamsBpOCL
	{
		float m_invCellSize[4];
		int   m_gridSize[4];
	};
	btParamsBpOCL hParams;
	hParams.m_invCellSize[0] = m_params.m_invCellSizeX;
	hParams.m_invCellSize[1] = m_params.m_invCellSizeY;
	hParams.m_invCellSize[2] = m_params.m_invCellSizeZ;
	hParams.m_invCellSize[3] = 0.f;
	hParams.m_gridSize[0] = m_params.m_gridSizeX;
	hParams.m_gridSize[1] = m_params.m_gridSizeY;
	hParams.m_gridSize[2] = m_params.m_gridSizeZ;
	hParams.m_gridSize[3] = m_params.m_maxBodiesPerCell;
	copyArrayToDevice(m_dBpParams, &hParams, sizeof(btParamsBpOCL));
	return;
}


void bt3dGridBroadphaseOCL::calcHashAABB()
{
	BT_PROFILE("calcHashAABB");
#if 1
	runKernelWithWorkgroupSize(GRID3DOCL_KERNEL_CALC_HASH_AABB, m_numHandles);
#else
	btGpu3DGridBroadphase::calcHashAABB();
#endif
	return;
}



void bt3dGridBroadphaseOCL::sortHash()
{
	BT_PROFILE("sortHash");
#if defined(CL_PLATFORM_MINI_CL) || defined(CL_PLATFORM_AMD)
	copyArrayFromDevice(m_hBodiesHash, m_dBodiesHash, m_numHandles * 2 * sizeof(unsigned int));
	btGpu3DGridBroadphase::sortHash();
	copyArrayToDevice(m_dBodiesHash, m_hBodiesHash, m_numHandles * 2 * sizeof(unsigned int));
#else
	int dir = 1;
	bitonicSort(m_dBodiesHash, m_hashSize, dir);
#endif
	return;
}



void bt3dGridBroadphaseOCL::findCellStart()
{
#if 1
	BT_PROFILE("findCellStart");
	runKernelWithWorkgroupSize(GRID3DOCL_KERNEL_CLEAR_CELL_START, m_numCells);
	#if defined(CL_PLATFORM_MINI_CL) || defined(CL_PLATFORM_AMD)
		btGpu3DGridBroadphase::findCellStart();
		copyArrayToDevice(m_dCellStart, m_hCellStart, m_numCells * sizeof(unsigned int));
	#else
		runKernelWithWorkgroupSize(GRID3DOCL_KERNEL_FIND_CELL_START, m_numHandles);
	#endif
#else
	btGpu3DGridBroadphase::findCellStart();
#endif
	return;
}



void bt3dGridBroadphaseOCL::findOverlappingPairs()
{
#if 1
	BT_PROFILE("findOverlappingPairs");
	runKernelWithWorkgroupSize(GRID3DOCL_KERNEL_FIND_OVERLAPPING_PAIRS, m_numHandles);
#else
	btGpu3DGridBroadphase::findOverlappingPairs();
#endif
	return;
}


void bt3dGridBroadphaseOCL::findPairsLarge()
{
	BT_PROFILE("findPairsLarge");
#if 1
	if(m_numLargeHandles)
	{
		setKernelArg(GRID3DOCL_KERNEL_FIND_PAIRS_LARGE, 6, sizeof(int),(void*)&m_numLargeHandles);
		runKernelWithWorkgroupSize(GRID3DOCL_KERNEL_FIND_PAIRS_LARGE, m_numHandles);
	}
#else
	btGpu3DGridBroadphase::findPairsLarge();
#endif
	return;
}



void bt3dGridBroadphaseOCL::computePairCacheChanges()
{
	BT_PROFILE("computePairCacheChanges");
#if 1
	runKernelWithWorkgroupSize(GRID3DOCL_KERNEL_COMPUTE_CACHE_CHANGES, m_numHandles);
#else
	btGpu3DGridBroadphase::computePairCacheChanges();
#endif
	return;
}

static unsigned int zeroEl = 0;


extern cl_device_type deviceType;

void bt3dGridBroadphaseOCL::scanOverlappingPairBuff()
{

	//Intel/CPU version doesn't handlel Adl scan well
#if 1
	{
		copyArrayFromDevice(m_hPairScan, m_dPairScan, sizeof(unsigned int)*(m_numHandles + 2)); 
		btGpu3DGridBroadphase::scanOverlappingPairBuff();
		copyArrayToDevice(m_dPairScan, m_hPairScan, sizeof(unsigned int)*(m_numHandles + 2)); 
	}
#else
	{
		adl::Buffer<unsigned int> destBuffer;
		
		destBuffer.m_ptr = (unsigned int*)m_dPairScan;
		destBuffer.m_device = m_deviceCL;
		destBuffer.m_size =  sizeof(unsigned int)*(m_numHandles+2);
		m_deviceCL->copy(m_srcClBuffer, &destBuffer,m_numHandles+1);
		static bool onlyOnce = true;
		if (onlyOnce)
		{
			onlyOnce = false;
			m_srcClBuffer->write(&zeroEl,1,m_numHandles+1);
		}
		//m_deviceCL->waitForCompletion();
		unsigned int gSum=0;
		adl::PrefixScan<adl::TYPE_CL>::execute(gData1,*m_srcClBuffer,destBuffer, m_numHandles+2,&gSum);
		//m_deviceCL->waitForCompletion();

		//the data 
		copyArrayFromDevice(m_hPairScan, m_dPairScan, sizeof(unsigned int)*(m_numHandles + 2));
	}
#endif

}



void bt3dGridBroadphaseOCL::squeezeOverlappingPairBuff()
{
	BT_PROFILE("btCuda_squeezeOverlappingPairBuff");
#if 1
	runKernelWithWorkgroupSize(GRID3DOCL_KERNEL_SQUEEZE_PAIR_BUFF, m_numHandles);
//	btCuda_squeezeOverlappingPairBuff(m_dPairBuff, m_dPairBuffStartCurr, m_dPairScan, m_dPairOut, m_dAABB, m_numHandles);
	
	copyArrayFromDevice(m_hPairOut, m_dPairOut, sizeof(unsigned int) * m_hPairScan[m_numHandles+1]); //gSum
	clFinish(m_cqCommandQue);
#else
	btGpu3DGridBroadphase::squeezeOverlappingPairBuff();
#endif
	return;
}



void bt3dGridBroadphaseOCL::resetPool(btDispatcher* dispatcher)
{
	btGpu3DGridBroadphase::resetPool(dispatcher);
	prefillBuffers();
}



//Note: logically shared with BitonicSort OpenCL code!

void bt3dGridBroadphaseOCL::bitonicSort(cl_mem pKey, unsigned int arrayLength, unsigned int dir)
{
	unsigned int localSizeLimit = m_kernels[GRID3DOCL_KERNEL_BITONIC_SORT_CELL_ID_LOCAL].m_workgroupSize * 2;
    if(arrayLength < 2)
        return;
    //Only power-of-two array lengths are supported so far
    dir = (dir != 0);
    cl_int ciErrNum;
    size_t localWorkSize, globalWorkSize;
    if(arrayLength <= localSizeLimit)
    {
        btAssert( (arrayLength % localSizeLimit) == 0);
        //Launch bitonicSortLocal
		ciErrNum  = clSetKernelArg(m_kernels[GRID3DOCL_KERNEL_BITONIC_SORT_CELL_ID_LOCAL].m_kernel, 0,   sizeof(cl_mem), (void *)&pKey);
        ciErrNum |= clSetKernelArg(m_kernels[GRID3DOCL_KERNEL_BITONIC_SORT_CELL_ID_LOCAL].m_kernel, 1,  sizeof(cl_uint), (void *)&arrayLength);
        ciErrNum |= clSetKernelArg(m_kernels[GRID3DOCL_KERNEL_BITONIC_SORT_CELL_ID_LOCAL].m_kernel, 2,  sizeof(cl_uint), (void *)&dir);
		GRID3DOCL_CHECKERROR(ciErrNum, CL_SUCCESS);

        localWorkSize  = localSizeLimit / 2;
        globalWorkSize = arrayLength / 2;
        ciErrNum = clEnqueueNDRangeKernel(m_cqCommandQue, m_kernels[GRID3DOCL_KERNEL_BITONIC_SORT_CELL_ID_LOCAL].m_kernel, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
		GRID3DOCL_CHECKERROR(ciErrNum, CL_SUCCESS);
    }
    else
    {
        //Launch bitonicSortLocal1
        ciErrNum  = clSetKernelArg(m_kernels[GRID3DOCL_KERNEL_BITONIC_SORT_CELL_ID_LOCAL_1].m_kernel, 0,  sizeof(cl_mem), (void *)&pKey);
		GRID3DOCL_CHECKERROR(ciErrNum, CL_SUCCESS);

        localWorkSize  = localSizeLimit / 2;
        globalWorkSize = arrayLength / 2;
        ciErrNum = clEnqueueNDRangeKernel(m_cqCommandQue, m_kernels[GRID3DOCL_KERNEL_BITONIC_SORT_CELL_ID_LOCAL_1].m_kernel, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
		GRID3DOCL_CHECKERROR(ciErrNum, CL_SUCCESS);

        for(unsigned int size = 2 * localSizeLimit; size <= arrayLength; size <<= 1)
        {
            for(unsigned stride = size / 2; stride > 0; stride >>= 1)
            {
                if(stride >= localSizeLimit)
                {
                    //Launch bitonicMergeGlobal
                    ciErrNum  = clSetKernelArg(m_kernels[GRID3DOCL_KERNEL_BITONIC_SORT_CELL_ID_MERGE_GLOBAL].m_kernel, 0,  sizeof(cl_mem), (void *)&pKey);
                    ciErrNum |= clSetKernelArg(m_kernels[GRID3DOCL_KERNEL_BITONIC_SORT_CELL_ID_MERGE_GLOBAL].m_kernel, 1, sizeof(cl_uint), (void *)&arrayLength);
                    ciErrNum |= clSetKernelArg(m_kernels[GRID3DOCL_KERNEL_BITONIC_SORT_CELL_ID_MERGE_GLOBAL].m_kernel, 2, sizeof(cl_uint), (void *)&size);
                    ciErrNum |= clSetKernelArg(m_kernels[GRID3DOCL_KERNEL_BITONIC_SORT_CELL_ID_MERGE_GLOBAL].m_kernel, 3, sizeof(cl_uint), (void *)&stride);
                    ciErrNum |= clSetKernelArg(m_kernels[GRID3DOCL_KERNEL_BITONIC_SORT_CELL_ID_MERGE_GLOBAL].m_kernel, 4, sizeof(cl_uint), (void *)&dir);
					GRID3DOCL_CHECKERROR(ciErrNum, CL_SUCCESS);

                    localWorkSize  = localSizeLimit / 4;
                    globalWorkSize = arrayLength / 2;

                    ciErrNum = clEnqueueNDRangeKernel(m_cqCommandQue, m_kernels[GRID3DOCL_KERNEL_BITONIC_SORT_CELL_ID_MERGE_GLOBAL].m_kernel, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
					GRID3DOCL_CHECKERROR(ciErrNum, CL_SUCCESS);
                }
                else
                {
                    //Launch bitonicMergeLocal
					ciErrNum  = clSetKernelArg(m_kernels[GRID3DOCL_KERNEL_BITONIC_SORT_CELL_ID_MERGE_LOCAL].m_kernel, 0,  sizeof(cl_mem), (void *)&pKey);
                    ciErrNum |= clSetKernelArg(m_kernels[GRID3DOCL_KERNEL_BITONIC_SORT_CELL_ID_MERGE_LOCAL].m_kernel, 1, sizeof(cl_uint), (void *)&arrayLength);
                    ciErrNum |= clSetKernelArg(m_kernels[GRID3DOCL_KERNEL_BITONIC_SORT_CELL_ID_MERGE_LOCAL].m_kernel, 2, sizeof(cl_uint), (void *)&stride);
                    ciErrNum |= clSetKernelArg(m_kernels[GRID3DOCL_KERNEL_BITONIC_SORT_CELL_ID_MERGE_LOCAL].m_kernel, 3, sizeof(cl_uint), (void *)&size);
                    ciErrNum |= clSetKernelArg(m_kernels[GRID3DOCL_KERNEL_BITONIC_SORT_CELL_ID_MERGE_LOCAL].m_kernel, 4, sizeof(cl_uint), (void *)&dir);
					GRID3DOCL_CHECKERROR(ciErrNum, CL_SUCCESS);

                    localWorkSize  = localSizeLimit / 2;
                    globalWorkSize = arrayLength / 2;

                    ciErrNum = clEnqueueNDRangeKernel(m_cqCommandQue, m_kernels[GRID3DOCL_KERNEL_BITONIC_SORT_CELL_ID_MERGE_LOCAL].m_kernel, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
					GRID3DOCL_CHECKERROR(ciErrNum, CL_SUCCESS);
                    break;
                }
            }
        }
    }
}

