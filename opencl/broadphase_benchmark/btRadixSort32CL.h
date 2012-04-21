
#ifndef BT_RADIXSORT32_H
#define BT_RADIXSORT32_H

#include "btOpenCLArray.h"

struct btSortData
{
	int m_key;
	int m_value;
};
#include "btBufferInfoCL.h"

struct btRadixSort32CL
{
	enum
		{
			DATA_ALIGNMENT = 256,
			WG_SIZE = 64,
			ELEMENTS_PER_WORK_ITEM = (256/WG_SIZE),
			BITS_PER_PASS = 4,

			//	if you change this, change nPerWI in kernel as well
			NUM_WGS = 20*6,	//	cypress
//			NUM_WGS = 24*6,	//	cayman
//			NUM_WGS = 32*4,	//	nv
		};


		struct btConstData
		{
			int m_n;
			int m_nWGs;
			int m_startBit;
			int m_nBlocksPerWG;
		};

		btOpenCLArray<unsigned int>* m_workBuffer1;
		
		btOpenCLArray<btSortData>*	m_workBuffer3;

		cl_command_queue	m_commandQueue;

		cl_kernel m_streamCountSortDataKernel;
		cl_kernel m_prefixScanKernel;
		cl_kernel m_sortAndScatterSortDataKernel;

		int m_maxSize;//??

		btOpenCLArray<btConstData>* m_constBuffer[32/BITS_PER_PASS];

		btRadixSort32CL(cl_context ctx, cl_device_id device, cl_command_queue queue, int initialCapacity =0);

		virtual ~btRadixSort32CL();


		void execute(btOpenCLArray<btSortData>& keyValuesInOut, int sortBits  = 32 );

};
#endif //BT_RADIXSORT32_H

