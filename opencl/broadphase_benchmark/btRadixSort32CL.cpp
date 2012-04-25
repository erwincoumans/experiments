
#include "btRadixSort32CL.h"
#include "btLauncherCL.h"
#include "../basic_initialize/btOpenCLUtils.h"
#include "btPrefixScanCL.h"
#include "btFillCL.h"

#ifdef _WIN32
#define RADIXSORT32_PATH "../../opencl/primitives/AdlPrimitives/Sort/RadixSort32Kernels.cl"
#else
#define RADIXSORT32_PATH "../opencl/primitives/AdlPrimitives/Sort/RadixSort32Kernels.cl"
#endif


btRadixSort32CL::btRadixSort32CL(cl_context ctx, cl_device_id device, cl_command_queue queue, int initialCapacity)
:m_commandQueue(queue)
{
	m_workBuffer1 = new btOpenCLArray<unsigned int>(ctx,queue);
	m_workBuffer2 = new btOpenCLArray<unsigned int>(ctx,queue);
	m_workBuffer3 = new btOpenCLArray<btSortData>(ctx,queue);


	if (initialCapacity>0)
	{
		m_workBuffer1->resize(initialCapacity);
		m_workBuffer3->resize(initialCapacity);
	}

	m_scan = new btPrefixScanCL(ctx,device,queue);
	m_fill = new btFillCL(ctx,device,queue);
	
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
	delete m_scan;
	delete m_fill;
	delete m_workBuffer1;
	delete m_workBuffer2;
	delete m_workBuffer3;

	clReleaseKernel(m_streamCountSortDataKernel);
	clReleaseKernel(m_sortAndScatterSortDataKernel);
	clReleaseKernel(m_prefixScanKernel);
}

void btRadixSort32CL::executeHost(btOpenCLArray<btSortData>& keyValuesInOut, int sortBits /* = 32 */)
{
	int n = keyValuesInOut.size();
	const int BITS_PER_PASS = 8;
	const int NUM_TABLES = (1<<BITS_PER_PASS);

	btAlignedObjectArray<btSortData> inout;
	keyValuesInOut.copyToHost(inout);

	int tables[NUM_TABLES];
	int counter[NUM_TABLES];

	btSortData* src = &inout[0];
	btAlignedObjectArray<btSortData> workbuffer;
	workbuffer.resize(inout.size());
	btSortData* dst = &workbuffer[0];

	int count=0;
	for(int startBit=0; startBit<sortBits; startBit+=BITS_PER_PASS)
	{
		for(int i=0; i<NUM_TABLES; i++)
		{
			tables[i] = 0;
		}

		for(int i=0; i<n; i++)
		{
			int tableIdx = (src[i].m_key >> startBit) & (NUM_TABLES-1);
			tables[tableIdx]++;
		}
#ifdef TEST
		printf("histogram size=%d\n",NUM_TABLES);
		for (int i=0;i<NUM_TABLES;i++)
		{
			if (tables[i]!=0)
			{
				printf("tables[%d]=%d]\n",i,tables[i]);
			}

		}
#endif //TEST
		//	prefix scan
		int sum = 0;
		for(int i=0; i<NUM_TABLES; i++)
		{
			int iData = tables[i];
			tables[i] = sum;
			sum += iData;
			counter[i] = 0;
		}

		//	distribute
		for(int i=0; i<n; i++)
		{
			int tableIdx = (src[i].m_key >> startBit) & (NUM_TABLES-1);
			
			dst[tables[tableIdx] + counter[tableIdx]] = src[i];
			counter[tableIdx] ++;
		}

		btSwap( src, dst );
		count++;
	}

	{
		if (count&1)
		//if( src != inout.m_ptr )
		{
			//memcpy( dst, src, sizeof(btSortData)*n );
			keyValuesInOut.copyFromHost(0,n,src);
		} else
		{
			keyValuesInOut.copyFromHost(0,n,dst);
		}
	}

}

void btRadixSort32CL::execute(btOpenCLArray<unsigned int>& keysIn, btOpenCLArray<unsigned int>& keysOut, btOpenCLArray<unsigned int>& valuesIn, 
								btOpenCLArray<unsigned int>& valuesOut, int n, int sortBits)
{

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
		btSortData fillValue;
		fillValue.m_key = UINT_MAX;//0xffffffff;

#define USE_BTFILL
#ifdef USE_BTFILL
		m_fill->execute((btOpenCLArray<btInt2>&)keyValuesInOut,(btInt2&)fillValue,workingSize-originalSize,originalSize);
#else
		//fill the remaining bits (very slow way, todo: fill on GPU/OpenCL side)
		
		for (int i=originalSize; i<workingSize;i++)
		{
			keyValuesInOut.copyFromHost(i, i+1, &fillValue);
		}
#endif//USE_BTFILL

	}
	
	
	btAssert( workingSize%DATA_ALIGNMENT == 0 );
	int minCap = NUM_BUCKET*NUM_WGS;

	if (safeSize<minCap)
	{
		safeSize = minCap;
	}

	int n = workingSize;

	m_workBuffer1->resize(safeSize);
	m_workBuffer3->resize(workingSize);
	

//	ADLASSERT( ELEMENTS_PER_WORK_ITEM == 4 );
	btAssert( BITS_PER_PASS == 4 );
	btAssert( WG_SIZE == 64 );
	btAssert( (sortBits&0x3) == 0 );

	
	btOpenCLArray<btSortData>* src = &keyValuesInOut;
	btOpenCLArray<btSortData>* dst = m_workBuffer3;

	btOpenCLArray<unsigned int>* srcHisto = m_workBuffer1;
	btOpenCLArray<unsigned int>* destHisto = m_workBuffer2;


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
			btBufferInfoCL bInfo[] = { btBufferInfoCL( src->getBufferCL(), true ), btBufferInfoCL( srcHisto->getBufferCL() ) };
			btLauncherCL launcher(m_commandQueue, m_streamCountSortDataKernel);

			launcher.setBuffers( bInfo, sizeof(bInfo)/sizeof(btBufferInfoCL) );
			launcher.setConst(  cdata );
			
			int num = NUM_WGS*WG_SIZE;
			launcher.launch1D( num, WG_SIZE );
		}
//#define DEBUG_RADIXSORT
#ifdef DEBUG_RADIXSORT
		btAlignedObjectArray<unsigned int> testHist;
		srcHisto->copyToHost(testHist);
		printf("ib = %d, testHist size = %d, non zero elements:\n",ib, testHist.size());
		for (int i=0;i<testHist.size();i++)
		{
			if (testHist[i]!=0)
				printf("testHist[%d]=%d\n",i,testHist[i]);
		}
#endif //DEBUG_RADIXSORT
	
		unsigned int sum;

//fast prefix scan is not working properly on Mac OSX yet
#ifdef _WIN32
	bool fastScan=true;
#else
	bool fastScan=false;
#endif

		if (fastScan)
		{//	prefix scan group histogram
			btBufferInfoCL bInfo[] = { btBufferInfoCL( srcHisto->getBufferCL() ) };
			btLauncherCL launcher( m_commandQueue, m_prefixScanKernel );
			launcher.setBuffers( bInfo, sizeof(bInfo)/sizeof(btBufferInfoCL) );
			launcher.setConst(  cdata );
			launcher.launch1D( 128, 128 );
			destHisto = srcHisto;
		}else
		{
			m_scan->execute(*srcHisto,*destHisto,srcHisto->size(),&sum);
		}


#ifdef DEBUG_RADIXSORT
		destHisto->copyToHost(testHist);
		printf("ib = %d, testHist size = %d, non zero elements:\n",ib, testHist.size());
		for (int i=0;i<testHist.size();i++)
		{
			if (testHist[i]!=0)
				printf("testHist[%d]=%d\n",i,testHist[i]);
		}
#endif //DEBUG_RADIXSORT
		{//	local sort and distribute
			btBufferInfoCL bInfo[] = { btBufferInfoCL( src->getBufferCL(), true ), btBufferInfoCL( destHisto->getBufferCL(), true ), btBufferInfoCL( dst->getBufferCL() )};
			btLauncherCL launcher( m_commandQueue, m_sortAndScatterSortDataKernel );
			launcher.setBuffers( bInfo, sizeof(bInfo)/sizeof(btBufferInfoCL) );
			launcher.setConst(  cdata );
			launcher.launch1D( nWGs*WG_SIZE, WG_SIZE );
		}
#ifdef DEBUG_RADIXSORT
		destHisto->copyToHost(testHist);
		printf("ib = %d, testHist size = %d, non zero elements:\n",ib, testHist.size());
		for (int i=0;i<testHist.size();i++)
		{
			if (testHist[i]!=0)
				printf("testHist[%d]=%d\n",i,testHist[i]);
		}
#endif //DEBUG_RADIXSORT
		btSwap(src, dst );
		btSwap(srcHisto,destHisto);

		count++;
	}
	
	if (count&1)
	{
		btAssert(0);//need to copy from workbuffer to keyValuesInOut
	}

	if (originalSize!=keyValuesInOut.size())
	{
		keyValuesInOut.resize(originalSize);
	}
}