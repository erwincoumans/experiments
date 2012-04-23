
#include "btRadixSort32CL.h"
#include "btLauncherCL.h"
#include "../basic_initialize/btOpenCLUtils.h"

#define RADIXSORT32_PATH "..\\..\\opencl\\primitives\\AdlPrimitives\\Sort\\RadixSort32Kernels.cl"
//#define PREFIXSCAN_PATH "..\\..\\opencl\\primitives\\AdlPrimitives\\Scan\\PrefixScanKernels.cl"


btRadixSort32CL::btRadixSort32CL(cl_context ctx, cl_device_id device, cl_command_queue queue, int initialCapacity)
:m_commandQueue(queue)
{
	m_workBuffer1 = new btOpenCLArray<unsigned int>(ctx,queue);
	m_workBuffer3 = new btOpenCLArray<btSortData>(ctx,queue);

	if (initialCapacity>0)
	{
		m_workBuffer1->resize(initialCapacity);
		m_workBuffer3->resize(initialCapacity);
	}

	
	const char* additionalMacros = "";
	const char* srcFileNameForCaching="";

	cl_int pErrNum;
	char* kernelSource = 0;
	
	cl_program sortProg = btOpenCLUtils::compileCLProgramFromString( ctx, device, kernelSource, &pErrNum,additionalMacros, RADIXSORT32_PATH);
	btAssert(sortProg);

	m_streamCountSortDataKernel = btOpenCLUtils::compileCLKernelFromString( ctx, device, kernelSource, "StreamCountSortDataKernel", &pErrNum, sortProg,additionalMacros );
	btAssert(m_streamCountSortDataKernel );
	m_sortAndScatterSortDataKernel = btOpenCLUtils::compileCLKernelFromString( ctx, device, kernelSource, "SortAndScatterSortDataKernel", &pErrNum, sortProg,additionalMacros );
	btAssert(m_sortAndScatterSortDataKernel);
	m_prefixScanKernel = btOpenCLUtils::compileCLKernelFromString( ctx, device, kernelSource, "PrefixScanKernel", &pErrNum, sortProg,additionalMacros );
	btAssert(m_prefixScanKernel);
		
}

btRadixSort32CL::~btRadixSort32CL()
{
	clReleaseKernel(m_streamCountSortDataKernel);
	clReleaseKernel(m_sortAndScatterSortDataKernel);
	clReleaseKernel(m_prefixScanKernel);
}


void btRadixSort32CL::execute(btOpenCLArray<btSortData>& keyValuesInOut, int sortBits /* = 32 */)
{
	
	int originalSize = keyValuesInOut.size();
	int workingSize = originalSize;
	
	int safeSize = originalSize;
		
	int dataAlignment = DATA_ALIGNMENT;

	if (workingSize%dataAlignment)
	{
		workingSize += dataAlignment-(workingSize%dataAlignment);
		safeSize = workingSize;
		keyValuesInOut.resize(workingSize);
		//fill the remaining bits (very slow way, todo: fill on GPU/OpenCL side)
		btSortData src;
		src.m_key = 0xffffffff;
		for (int i=originalSize; i<workingSize;i++)
		{
			keyValuesInOut.copyFromHost(i, i+1, &src);
		}
//		keyValuesInOut.
	}
	
	
	btAssert( workingSize%DATA_ALIGNMENT == 0 );
	
	int minCap = 256*1024;

	if (safeSize<minCap)
	{
		safeSize = minCap;
	}
	int n = workingSize;

	m_workBuffer1->resize(safeSize);
	m_workBuffer3->resize(safeSize);

	

//	ADLASSERT( ELEMENTS_PER_WORK_ITEM == 4 );
	btAssert( BITS_PER_PASS == 4 );
	btAssert( WG_SIZE == 64 );
	btAssert( (sortBits&0x3) == 0 );

	
	btOpenCLArray<btSortData>* src = &keyValuesInOut;
	btOpenCLArray<btSortData>* dst = m_workBuffer3;

	btOpenCLArray<unsigned int>* histogramBuffer = m_workBuffer1;


	int nWGs = NUM_WGS;
	btConstData cdata;

	{
		int nBlocks = (n+ELEMENTS_PER_WORK_ITEM*WG_SIZE-1)/(ELEMENTS_PER_WORK_ITEM*WG_SIZE);
		cdata.m_n = n;
		cdata.m_nWGs = NUM_WGS;
		cdata.m_startBit = 0;
		cdata.m_nBlocksPerWG = (nBlocks + cdata.m_nWGs - 1)/cdata.m_nWGs;
		if( nBlocks < NUM_WGS )
		{
			cdata.m_nBlocksPerWG = 1;
			nWGs = nBlocks;
		}
	}

	int count=0;
	for(int ib=0; ib<sortBits; ib+=4)
	{
		cdata.m_startBit = ib;
		
		{
			btBufferInfoCL bInfo[] = { btBufferInfoCL( src->getBufferCL(), true ), btBufferInfoCL( histogramBuffer->getBufferCL() ) };
			btLauncherCL launcher(m_commandQueue, m_streamCountSortDataKernel);

			launcher.setBuffers( bInfo, sizeof(bInfo)/sizeof(btBufferInfoCL) );
			launcher.setConst(  cdata );
			
			launcher.launch1D( NUM_WGS*WG_SIZE, WG_SIZE );
		}
		
		{//	prefix scan group histogram
			btBufferInfoCL bInfo[] = { btBufferInfoCL( histogramBuffer->getBufferCL() ) };
			btLauncherCL launcher( m_commandQueue, m_prefixScanKernel );
			launcher.setBuffers( bInfo, sizeof(bInfo)/sizeof(btBufferInfoCL) );
			launcher.setConst(  cdata );
			launcher.launch1D( 128, 128 );
		}
		{//	local sort and distribute
			btBufferInfoCL bInfo[] = { btBufferInfoCL( src->getBufferCL(), true ), btBufferInfoCL( histogramBuffer->getBufferCL(), true ), btBufferInfoCL( dst->getBufferCL() )};
			btLauncherCL launcher( m_commandQueue, m_sortAndScatterSortDataKernel );
			launcher.setBuffers( bInfo, sizeof(bInfo)/sizeof(btBufferInfoCL) );
			launcher.setConst(  cdata );
			launcher.launch1D( nWGs*WG_SIZE, WG_SIZE );
		}
		

		btSwap(src, dst );
		count++;
	}
	
	if (count&1)
	{
		btAssert(0);//need to copy from workbuffer to keyValuesInOut
	}

	if (originalSize<minCap)
	{
		keyValuesInOut.resize(originalSize);
	}
}