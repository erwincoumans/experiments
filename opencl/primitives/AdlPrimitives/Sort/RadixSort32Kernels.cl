

#pragma OPENCL EXTENSION cl_amd_printf : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

typedef unsigned int u32;
#define GET_GROUP_IDX get_group_id(0)
#define GET_LOCAL_IDX get_local_id(0)
#define GET_GLOBAL_IDX get_global_id(0)
#define GET_GROUP_SIZE get_local_size(0)
#define GROUP_LDS_BARRIER barrier(CLK_LOCAL_MEM_FENCE)
#define GROUP_MEM_FENCE mem_fence(CLK_LOCAL_MEM_FENCE)
#define AtomInc(x) atom_inc(&(x))
#define AtomInc1(x, out) out = atom_inc(&(x))
#define AtomAdd(x, value) atom_add(&(x), value)

#define SELECT_UINT4( b, a, condition ) select( b,a,condition )


#define make_uint4 (uint4)
#define make_uint2 (uint2)
#define make_int2 (int2)

#define WG_SIZE 64
#define ELEMENTS_PER_WORK_ITEM 4
#define BITS_PER_PASS 4
#define NUM_BUCKET (1<<BITS_PER_PASS)
typedef uchar u8;

//	this isn't optimization for VLIW. But just reducing writes. 
#define USE_2LEVEL_REDUCE 1


typedef struct
{
	int m_n;
	int m_nWGs;
	int m_startBit;
	int m_nBlocksPerWG;
} ConstBuffer;


u32 prefixScanEx( u32* data )
{
	u32 sum = 0;
	u32 tmp = data[0];
	data[0] = sum;
	sum += tmp;
	tmp = data[1];
	data[1] = sum;
	sum += tmp;
	tmp = data[2];
	data[2] = sum;
	sum += tmp;
	tmp = data[3];
	data[3] = sum;
	sum += tmp;
	return sum;
}


uint prefixScanVectorEx( uint4* data )
{
/*
	uint4 backup = data[0];
	data[0].y += data[0].x;
	data[0].w += data[0].z;
	data[0].z += data[0].y;
	data[0].w += data[0].y;
	uint sum = data[0].w;
	data[0] -= backup;
	return sum;
*/
	u32 sum = 0;
	u32 tmp = data[0].x;
	data[0].x = sum;
	sum += tmp;
	tmp = data[0].y;
	data[0].y = sum;
	sum += tmp;
	tmp = data[0].z;
	data[0].z = sum;
	sum += tmp;
	tmp = data[0].w;
	data[0].w = sum;
	sum += tmp;
	return sum;
}

/*
//__attribute__((reqd_work_group_size(128,1,1)))
u32 localPrefixSum128VSingle( u32 pData, uint lIdx, uint* totalSum, __local u32 sorterSharedMemory[] )
{
	{	//	Set data
		sorterSharedMemory[lIdx] = 0;
		sorterSharedMemory[lIdx+WG_SIZE] = pData;
	}

	GROUP_LDS_BARRIER;

	{	//	Prefix sum
		int idx = 2*lIdx + (WG_SIZE+1);
		if( lIdx < 64 )
		{
			sorterSharedMemory[idx] += sorterSharedMemory[idx-1];
			GROUP_MEM_FENCE;
			sorterSharedMemory[idx] += sorterSharedMemory[idx-2];			
			GROUP_MEM_FENCE;
			sorterSharedMemory[idx] += sorterSharedMemory[idx-4];
			GROUP_MEM_FENCE;
			sorterSharedMemory[idx] += sorterSharedMemory[idx-8];
			GROUP_MEM_FENCE;
			sorterSharedMemory[idx] += sorterSharedMemory[idx-16];
			GROUP_MEM_FENCE;
			sorterSharedMemory[idx] += sorterSharedMemory[idx-32];
			GROUP_MEM_FENCE;
			sorterSharedMemory[idx] += sorterSharedMemory[idx-64];
			GROUP_MEM_FENCE;

			sorterSharedMemory[idx-1] += sorterSharedMemory[idx-2];
			GROUP_MEM_FENCE;
		}
	}

	GROUP_LDS_BARRIER;

	*totalSum = sorterSharedMemory[WG_SIZE*2-1];
	u32 addValue = sorterSharedMemory[lIdx+127];
	return addValue;
}
*/
//__attribute__((reqd_work_group_size(64,1,1)))
u32 localPrefixSum64VSingle( u32 pData, uint lIdx, uint* totalSum, __local u32 sorterSharedMemory[] )
{
	const int wgSize = 64;
	{	//	Set data
		sorterSharedMemory[lIdx] = 0;
		sorterSharedMemory[lIdx+wgSize] = pData;
	}

	GROUP_LDS_BARRIER;

	{	//	Prefix sum
		int idx = 2*lIdx + (wgSize+1);
#if defined(USE_2LEVEL_REDUCE)
		if( lIdx < 64 )
		{
			u32 u0, u1, u2;
			u0 = sorterSharedMemory[idx-3];
			u1 = sorterSharedMemory[idx-2];
			u2 = sorterSharedMemory[idx-1];
			AtomAdd( sorterSharedMemory[idx], u0+u1+u2 );			
			GROUP_MEM_FENCE;

			u0 = sorterSharedMemory[idx-12];
			u1 = sorterSharedMemory[idx-8];
			u2 = sorterSharedMemory[idx-4];
			AtomAdd( sorterSharedMemory[idx], u0+u1+u2 );			
			GROUP_MEM_FENCE;

			u0 = sorterSharedMemory[idx-48];
			u1 = sorterSharedMemory[idx-32];
			u2 = sorterSharedMemory[idx-16];
			AtomAdd( sorterSharedMemory[idx], u0+u1+u2 );			
			GROUP_MEM_FENCE;

			sorterSharedMemory[idx-1] += sorterSharedMemory[idx-2];
			GROUP_MEM_FENCE;
		}
#else
		if( lIdx < 64 )
		{
			sorterSharedMemory[idx] += sorterSharedMemory[idx-1];
			GROUP_MEM_FENCE;
			sorterSharedMemory[idx] += sorterSharedMemory[idx-2];			
			GROUP_MEM_FENCE;

			sorterSharedMemory[idx] += sorterSharedMemory[idx-4];
			GROUP_MEM_FENCE;
			sorterSharedMemory[idx] += sorterSharedMemory[idx-8];
			GROUP_MEM_FENCE;

			sorterSharedMemory[idx] += sorterSharedMemory[idx-16];
			GROUP_MEM_FENCE;
			sorterSharedMemory[idx] += sorterSharedMemory[idx-32];
			GROUP_MEM_FENCE;
//			sorterSharedMemory[idx] += sorterSharedMemory[idx-64];
//			GROUP_MEM_FENCE;

			sorterSharedMemory[idx-1] += sorterSharedMemory[idx-2];
			GROUP_MEM_FENCE;
		}
#endif
	}

	GROUP_LDS_BARRIER;

	*totalSum = sorterSharedMemory[wgSize*2-1];
	u32 addValue = sorterSharedMemory[lIdx+wgSize-1];
	return addValue;
}

//__attribute__((reqd_work_group_size(128,1,1)))
uint4 localPrefixSum128V( uint4 pData, uint lIdx, uint* totalSum, __local u32 sorterSharedMemory[] )
{
	const int wgSize = 128;
	{	//	Set data
		sorterSharedMemory[lIdx] = 0;
		sorterSharedMemory[lIdx+wgSize] = prefixScanVectorEx( &pData );
	}

	GROUP_LDS_BARRIER;

	{	//	Prefix sum
		int idx = 2*lIdx + (wgSize+1);
		if( lIdx < 64 )
		{
			sorterSharedMemory[idx] += sorterSharedMemory[idx-1];
			GROUP_MEM_FENCE;
			sorterSharedMemory[idx] += sorterSharedMemory[idx-2];			
			GROUP_MEM_FENCE;
			sorterSharedMemory[idx] += sorterSharedMemory[idx-4];
			GROUP_MEM_FENCE;
			sorterSharedMemory[idx] += sorterSharedMemory[idx-8];
			GROUP_MEM_FENCE;
			sorterSharedMemory[idx] += sorterSharedMemory[idx-16];
			GROUP_MEM_FENCE;
			sorterSharedMemory[idx] += sorterSharedMemory[idx-32];
			GROUP_MEM_FENCE;
			sorterSharedMemory[idx] += sorterSharedMemory[idx-64];
			GROUP_MEM_FENCE;

			sorterSharedMemory[idx-1] += sorterSharedMemory[idx-2];
			GROUP_MEM_FENCE;
		}
	}

	GROUP_LDS_BARRIER;

	*totalSum = sorterSharedMemory[wgSize*2-1];
	uint addValue = sorterSharedMemory[lIdx+127];
	return pData + make_uint4(addValue, addValue, addValue, addValue);
}


//__attribute__((reqd_work_group_size(64,1,1)))
uint4 localPrefixSum64V( uint4 pData, uint lIdx, uint* totalSum, __local u32 sorterSharedMemory[] )
{
	u32 s4 = prefixScanVectorEx( &pData );
	u32 rank = localPrefixSum64VSingle( s4, lIdx, totalSum, sorterSharedMemory );
	return pData + make_uint4( rank, rank, rank, rank );
}

//===

u32 unpack4Key( u32 key, int keyIdx ){ return (key>>(keyIdx*8)) & 0xff;}

uint4 extractKeys(uint4 data, uint targetKey)
{
	uint4 key;
	key.x = data.x == targetKey ? 1:0;
	key.y = data.y == targetKey ? 1:0;
	key.z = data.z == targetKey ? 1:0;
	key.w = data.w == targetKey ? 1:0;
	return key;
}

//===




#define MY_HISTOGRAM(idx) localHistogramMat[(idx)*WG_SIZE+lIdx]


__kernel
__attribute__((reqd_work_group_size(WG_SIZE,1,1)))
void StreamCountKernel( __global u32* gSrc, __global u32* histogramOut, ConstBuffer cb )
{
	__local u32 localHistogramMat[NUM_BUCKET*WG_SIZE];

	u32 gIdx = GET_GLOBAL_IDX;
	u32 lIdx = GET_LOCAL_IDX;
	u32 wgIdx = GET_GROUP_IDX;
	u32 wgSize = GET_GROUP_SIZE;
	const int startBit = cb.m_startBit;

	const int n = cb.m_n;
	const int nWGs = cb.m_nWGs;
	const int nBlocksPerWG = cb.m_nBlocksPerWG;

	for(int i=0; i<NUM_BUCKET; i++)
	{
		MY_HISTOGRAM(i) = 0;
	}

	GROUP_LDS_BARRIER;

	const int blockSize = ELEMENTS_PER_WORK_ITEM*WG_SIZE;
	u32 localKeys[ELEMENTS_PER_WORK_ITEM];
//	for(int addr=ELEMENTS_PER_WORK_ITEM*gIdx; addr<n; addr+=stride)
	for(int addr = blockSize*nBlocksPerWG*wgIdx+ELEMENTS_PER_WORK_ITEM*lIdx; 
		addr<min(blockSize*nBlocksPerWG*(wgIdx+1), n); 
		addr+=blockSize )
	{
		for(int i=0; i<ELEMENTS_PER_WORK_ITEM; i++)
			localKeys[i] = (gSrc[addr+i]>>startBit) & 0xf;

		//	MY_HISTOGRAM( localKeys.x ) ++ is much expensive than atomic add as it requires read and write while atomics can just add
		//	Using registers didn't perform well. It seems like use localKeys to address requires a lot of alu ops
		for(int i=0; i<ELEMENTS_PER_WORK_ITEM; i++)
			AtomInc( MY_HISTOGRAM( localKeys[i] ) );
	}

	if( lIdx < NUM_BUCKET )
	{
		u32 sum = 0;
		for(int i=0; i<GET_GROUP_SIZE; i++)
		{
			sum += localHistogramMat[lIdx*WG_SIZE+i];
		}
		histogramOut[lIdx*nWGs+wgIdx] = sum;
//		histogramOut[wgIdx*NUM_BUCKET+lIdx] = sum;
	}
}


#define nPerWI 16
#define nPerLane (nPerWI/4)

//	NUM_BUCKET*nWGs < 128*nPerWI
__kernel
__attribute__((reqd_work_group_size(128,1,1)))
void PrefixScanKernel( __global u32* wHistogram1, ConstBuffer cb )
{
	__local u32 ldsSortData[128*2];

	u32 lIdx = GET_LOCAL_IDX;
	u32 wgIdx = GET_GROUP_IDX;
	const int nWGs = cb.m_nWGs;

	u32 data[nPerWI];
	for(int i=0; i<nPerWI; i++)
	{
		data[i] = 0;
		if( (nPerWI*lIdx+i) < NUM_BUCKET*nWGs )
			data[i] = wHistogram1[nPerWI*lIdx+i];
	}

	uint4 myData = make_uint4(0,0,0,0);

	for(int i=0; i<nPerLane; i++)
	{
		myData.x += data[nPerLane*0+i];
		myData.y += data[nPerLane*1+i];
		myData.z += data[nPerLane*2+i];
		myData.w += data[nPerLane*3+i];
	}

	uint totalSum;
	uint4 scanned = localPrefixSum128V( myData, lIdx, &totalSum, ldsSortData );

//	for(int j=0; j<4; j++) //	somehow it introduces a lot of branches
	{	int j = 0;
		u32 sum = 0;
		for(int i=0; i<nPerLane; i++)
		{
			u32 tmp = data[nPerLane*j+i];
			data[nPerLane*j+i] = sum;
			sum += tmp;
		}
	}
	{	int j = 1;
		u32 sum = 0;
		for(int i=0; i<nPerLane; i++)
		{
			u32 tmp = data[nPerLane*j+i];
			data[nPerLane*j+i] = sum;
			sum += tmp;
		}
	}
	{	int j = 2;
		u32 sum = 0;
		for(int i=0; i<nPerLane; i++)
		{
			u32 tmp = data[nPerLane*j+i];
			data[nPerLane*j+i] = sum;
			sum += tmp;
		}
	}
	{	int j = 3;
		u32 sum = 0;
		for(int i=0; i<nPerLane; i++)
		{
			u32 tmp = data[nPerLane*j+i];
			data[nPerLane*j+i] = sum;
			sum += tmp;
		}
	}

	for(int i=0; i<nPerLane; i++)
	{
		data[nPerLane*0+i] += scanned.x;
		data[nPerLane*1+i] += scanned.y;
		data[nPerLane*2+i] += scanned.z;
		data[nPerLane*3+i] += scanned.w;
	}

	for(int i=0; i<nPerWI; i++)
	{
		wHistogram1[nPerWI*lIdx+i] = data[i];
	}
}

void sort4Bits(u32 sortData[4], int startBit, int lIdx, __local u32 ldsSortData[WG_SIZE*ELEMENTS_PER_WORK_ITEM+16])
{
	for(int bitIdx=0; bitIdx<BITS_PER_PASS; bitIdx++)
	{
		u32 mask = (1<<bitIdx);
		uint4 cmpResult = make_uint4( (sortData[0]>>startBit) & mask, (sortData[1]>>startBit) & mask, (sortData[2]>>startBit) & mask, (sortData[3]>>startBit) & mask );
		uint4 prefixSum = SELECT_UINT4( make_uint4(1,1,1,1), make_uint4(0,0,0,0), cmpResult != make_uint4(0,0,0,0) );
		u32 total;
		prefixSum = localPrefixSum64V( prefixSum, lIdx, &total, ldsSortData );
//		prefixSum = localPrefixSum128V( prefixSum, lIdx, &total, ldsSortData );

		{
			uint4 localAddr = make_uint4(lIdx*4+0,lIdx*4+1,lIdx*4+2,lIdx*4+3);
			uint4 dstAddr = localAddr - prefixSum + make_uint4( total, total, total, total );
			dstAddr = SELECT_UINT4( prefixSum, dstAddr, cmpResult != make_uint4(0, 0, 0, 0) );

			GROUP_LDS_BARRIER;

			ldsSortData[dstAddr.x] = sortData[0];
			ldsSortData[dstAddr.y] = sortData[1];
			ldsSortData[dstAddr.z] = sortData[2];
			ldsSortData[dstAddr.w] = sortData[3];

			GROUP_LDS_BARRIER;

			sortData[0] = ldsSortData[localAddr.x];
			sortData[1] = ldsSortData[localAddr.y];
			sortData[2] = ldsSortData[localAddr.z];
			sortData[3] = ldsSortData[localAddr.w];

			GROUP_LDS_BARRIER;
		}
	}
}

void sort4Bits1(u32 sortData[4], int startBit, int lIdx, __local u32* ldsSortData)
{
	for(uint ibit=0; ibit<BITS_PER_PASS; ibit+=2)
	{
		uint4 b = make_uint4((sortData[0]>>(startBit+ibit)) & 0x3, 
			(sortData[1]>>(startBit+ibit)) & 0x3, 
			(sortData[2]>>(startBit+ibit)) & 0x3, 
			(sortData[3]>>(startBit+ibit)) & 0x3);

		u32 key4;
		u32 sKeyPacked[4] = { 0, 0, 0, 0 };
		{
			sKeyPacked[0] |= 1<<(8*b.x);
			sKeyPacked[1] |= 1<<(8*b.y);
			sKeyPacked[2] |= 1<<(8*b.z);
			sKeyPacked[3] |= 1<<(8*b.w);

			key4 = sKeyPacked[0] + sKeyPacked[1] + sKeyPacked[2] + sKeyPacked[3];
		}

		u32 rankPacked;
		u32 sumPacked;
		{
			rankPacked = localPrefixSum64VSingle( key4, lIdx, &sumPacked, ldsSortData );
		}

		GROUP_LDS_BARRIER;

		u32 sum[4] = { unpack4Key( sumPacked,0 ), unpack4Key( sumPacked,1 ), unpack4Key( sumPacked,2 ), unpack4Key( sumPacked,3 ) };

		{
			u32 sum4 = 0;
			for(int ie=0; ie<4; ie++)
			{
				u32 tmp = sum[ie];
				sum[ie] = sum4;
				sum4 += tmp;
			}
		}

		u32 newOffset[4] = { 0,0,0,0 };

		for(int ie=0; ie<4; ie++)
		{
			uint4 key = extractKeys( b, ie );
			uint4 scannedKey = key;
			prefixScanVectorEx( &scannedKey );
			uint offset = sum[ie] + unpack4Key( rankPacked, ie );
			uint4 dstAddress = make_uint4( offset, offset, offset, offset ) + scannedKey;

			newOffset[0] += dstAddress.x*key.x;
			newOffset[1] += dstAddress.y*key.y;
			newOffset[2] += dstAddress.z*key.z;
			newOffset[3] += dstAddress.w*key.w;
		}



		{
			ldsSortData[newOffset[0]] = sortData[0];
			ldsSortData[newOffset[1]] = sortData[1];
			ldsSortData[newOffset[2]] = sortData[2];
			ldsSortData[newOffset[3]] = sortData[3];

			GROUP_LDS_BARRIER;

			sortData[0] = ldsSortData[lIdx*4+0];
			sortData[1] = ldsSortData[lIdx*4+1];
			sortData[2] = ldsSortData[lIdx*4+2];
			sortData[3] = ldsSortData[lIdx*4+3];

			GROUP_LDS_BARRIER;
		}
	}
}

__kernel
__attribute__((reqd_work_group_size(WG_SIZE,1,1)))
void SortAndScatterKernel( __global u32* restrict gSrc, __global u32* rHistogram, __global u32* restrict gDst, ConstBuffer cb )
{
	__local u32 ldsSortData[WG_SIZE*ELEMENTS_PER_WORK_ITEM+16];
	__local u32 localHistogramToCarry[NUM_BUCKET];
	__local u32 localHistogram[NUM_BUCKET*2];


	u32 gIdx = GET_GLOBAL_IDX;
	u32 lIdx = GET_LOCAL_IDX;
	u32 wgIdx = GET_GROUP_IDX;
	u32 wgSize = GET_GROUP_SIZE;

	const int n = cb.m_n;
	const int nWGs = cb.m_nWGs;
	const int startBit = cb.m_startBit;
	const int nBlocksPerWG = cb.m_nBlocksPerWG;

	if( lIdx < (NUM_BUCKET) )
	{
		localHistogramToCarry[lIdx] = rHistogram[lIdx*nWGs + wgIdx];
	}

	GROUP_LDS_BARRIER;

	const int blockSize = ELEMENTS_PER_WORK_ITEM*WG_SIZE;
//	for(int addr=ELEMENTS_PER_WORK_ITEM*gIdx; addr<n; addr+=stride)
	for(int addr = blockSize*nBlocksPerWG*wgIdx+ELEMENTS_PER_WORK_ITEM*lIdx; 
		addr<min(blockSize*nBlocksPerWG*(wgIdx+1), n); 
		addr+=blockSize )
	{
		u32 myHistogram = 0;

		u32 sortData[ELEMENTS_PER_WORK_ITEM];
		{
			for(int i=0; i<ELEMENTS_PER_WORK_ITEM; i++)
				sortData[i] = gSrc[ addr+i ];
		}

		sort4Bits1(sortData, startBit, lIdx, ldsSortData);

		u32 keys[4];
		for(int i=0; i<4; i++)
			keys[i] = (sortData[i]>>startBit) & 0xf;

		{	//	create histogram
			if( lIdx < NUM_BUCKET )
			{
				localHistogram[lIdx] = 0;
				localHistogram[NUM_BUCKET+lIdx] = 0;
			}
			GROUP_LDS_BARRIER;

			AtomInc( localHistogram[NUM_BUCKET+keys[0]] );
			AtomInc( localHistogram[NUM_BUCKET+keys[1]] );
			AtomInc( localHistogram[NUM_BUCKET+keys[2]] );
			AtomInc( localHistogram[NUM_BUCKET+keys[3]] );
			
			GROUP_LDS_BARRIER;
			
			uint hIdx = NUM_BUCKET+lIdx;
			if( lIdx < NUM_BUCKET )
			{
				myHistogram = localHistogram[hIdx];
			}
			GROUP_LDS_BARRIER;

#if defined(USE_2LEVEL_REDUCE)
			if( lIdx < NUM_BUCKET )
			{
				localHistogram[hIdx] = localHistogram[hIdx-1];
				GROUP_MEM_FENCE;

				u32 u0, u1, u2;
				u0 = localHistogram[hIdx-3];
				u1 = localHistogram[hIdx-2];
				u2 = localHistogram[hIdx-1];
				AtomAdd( localHistogram[hIdx], u0 + u1 + u2 );
				GROUP_MEM_FENCE;
				u0 = localHistogram[hIdx-12];
				u1 = localHistogram[hIdx-8];
				u2 = localHistogram[hIdx-4];
				AtomAdd( localHistogram[hIdx], u0 + u1 + u2 );
				GROUP_MEM_FENCE;
			}
#else
			if( lIdx < NUM_BUCKET )
			{
				localHistogram[hIdx] = localHistogram[hIdx-1];
				GROUP_MEM_FENCE;
				localHistogram[hIdx] += localHistogram[hIdx-1];
				GROUP_MEM_FENCE;
				localHistogram[hIdx] += localHistogram[hIdx-2];
				GROUP_MEM_FENCE;
				localHistogram[hIdx] += localHistogram[hIdx-4];
				GROUP_MEM_FENCE;
				localHistogram[hIdx] += localHistogram[hIdx-8];
				GROUP_MEM_FENCE;
			}
#endif

			GROUP_LDS_BARRIER;
		}

		{
			for(int ie=0; ie<ELEMENTS_PER_WORK_ITEM; ie++)
			{
				int dataIdx = 4*lIdx+ie;
				int binIdx = keys[ie];
				int groupOffset = localHistogramToCarry[binIdx];
				int myIdx = dataIdx - localHistogram[NUM_BUCKET+binIdx];
				gDst[ groupOffset + myIdx ] = sortData[ie];
			}
		}

		GROUP_LDS_BARRIER;

		if( lIdx < NUM_BUCKET )
		{
			localHistogramToCarry[lIdx] += myHistogram;
		}
		GROUP_LDS_BARRIER;

	}
}

