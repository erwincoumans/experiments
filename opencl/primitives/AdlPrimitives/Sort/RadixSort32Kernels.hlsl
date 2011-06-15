/*
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2011 Advanced Micro Devices, Inc.  http://bulletphysics.org

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/
//Author Takahiro Harada

typedef uint u32;

#define GET_GROUP_IDX groupIdx.x
#define GET_LOCAL_IDX localIdx.x
#define GET_GLOBAL_IDX globalIdx.x
#define GROUP_LDS_BARRIER GroupMemoryBarrierWithGroupSync()
#define GROUP_MEM_FENCE
#define DEFAULT_ARGS uint3 globalIdx : SV_DispatchThreadID, uint3 localIdx : SV_GroupThreadID, uint3 groupIdx : SV_GroupID
#define AtomInc(x) InterlockedAdd(x, 1)
#define AtomInc1(x, out) InterlockedAdd(x, 1, out)
#define AtomAdd(x, inc) InterlockedAdd(x, inc)

#define make_uint4 uint4
#define make_uint2 uint2


#define WG_SIZE 64
#define ELEMENTS_PER_WORK_ITEM 4
#define BITS_PER_PASS 4
#define NUM_BUCKET (1<<BITS_PER_PASS)

//	this isn't optimization for VLIW. But just reducing writes. 
#define USE_2LEVEL_REDUCE 1


#define GET_GROUP_SIZE WG_SIZE


cbuffer SortCB : register( b0 )
{
	int m_n;
	int m_nWGs;
	int m_startBit;
	int m_nBlocksPerWG;
};


StructuredBuffer<u32> gSrc : register( t0 );
RWStructuredBuffer<u32> histogramOut : register( u0 );


groupshared u32 localHistogramMat[NUM_BUCKET*WG_SIZE];

#define MY_HISTOGRAM(idx) localHistogramMat[(idx)*WG_SIZE+lIdx]


[numthreads(WG_SIZE, 1, 1)]
void StreamCountKernel( DEFAULT_ARGS )
{
	u32 gIdx = GET_GLOBAL_IDX;
	u32 lIdx = GET_LOCAL_IDX;
	u32 wgIdx = GET_GROUP_IDX;
	u32 wgSize = GET_GROUP_SIZE;
	const int startBit = m_startBit;

	const int n = m_n;
	const int nWGs = m_nWGs;
	const int nBlocksPerWG = m_nBlocksPerWG;

	for(int i=0; i<NUM_BUCKET; i++)
	{
		MY_HISTOGRAM(i) = 0;
	}

	GROUP_LDS_BARRIER;

	const int blockSize = ELEMENTS_PER_WORK_ITEM*WG_SIZE;
	u32 localKeys[ELEMENTS_PER_WORK_ITEM];
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




uint prefixScanVectorEx( inout uint4 data )
{
	u32 sum = 0;
	u32 tmp = data.x;
	data.x = sum;
	sum += tmp;
	tmp = data.y;
	data.y = sum;
	sum += tmp;
	tmp = data.z;
	data.z = sum;
	sum += tmp;
	tmp = data.w;
	data.w = sum;
	sum += tmp;
	return sum;
}


groupshared u32 ldsSortData1[128*2];


//__attribute__((reqd_work_group_size(128,1,1)))
uint4 localPrefixSum128V( uint4 pData, uint lIdx, inout uint totalSum )
{
	const int wgSize = 128;
	{	//	Set data
		ldsSortData1[lIdx] = 0;
		ldsSortData1[lIdx+wgSize] = prefixScanVectorEx( pData );
	}

	GROUP_LDS_BARRIER;

	{	//	Prefix sum
		int idx = 2*lIdx + (wgSize+1);
		if( lIdx < 64 )
		{
			ldsSortData1[idx] += ldsSortData1[idx-1];
			GROUP_MEM_FENCE;
			ldsSortData1[idx] += ldsSortData1[idx-2];			
			GROUP_MEM_FENCE;
			ldsSortData1[idx] += ldsSortData1[idx-4];
			GROUP_MEM_FENCE;
			ldsSortData1[idx] += ldsSortData1[idx-8];
			GROUP_MEM_FENCE;
			ldsSortData1[idx] += ldsSortData1[idx-16];
			GROUP_MEM_FENCE;
			ldsSortData1[idx] += ldsSortData1[idx-32];
			GROUP_MEM_FENCE;
			ldsSortData1[idx] += ldsSortData1[idx-64];
			GROUP_MEM_FENCE;

			ldsSortData1[idx-1] += ldsSortData1[idx-2];
			GROUP_MEM_FENCE;
		}
	}

	GROUP_LDS_BARRIER;

	totalSum = ldsSortData1[wgSize*2-1];
	uint addValue = ldsSortData1[lIdx+127];
	return pData + make_uint4(addValue, addValue, addValue, addValue);
}


RWStructuredBuffer<u32> wHistogram1 : register( u0 );


#define nPerWI 16
#define nPerLane (nPerWI/4)

[numthreads(128, 1, 1)]
void PrefixScanKernel( DEFAULT_ARGS )
{
	u32 lIdx = GET_LOCAL_IDX;
	u32 wgIdx = GET_GROUP_IDX;
	const int nWGs = m_nWGs;

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
	uint4 scanned = localPrefixSum128V( myData, lIdx, totalSum );

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





groupshared u32 ldsSortData[WG_SIZE*ELEMENTS_PER_WORK_ITEM+16];

u32 localPrefixSum64VSingle( u32 pData, uint lIdx, inout uint totalSum )
{
	const int wgSize = 64;
	{	//	Set data
		ldsSortData[lIdx] = 0;
		ldsSortData[lIdx+wgSize] = pData;
	}

//	GROUP_LDS_BARRIER;

	{	//	Prefix sum
		int idx = 2*lIdx + (wgSize+1);
#if defined(USE_2LEVEL_REDUCE)
		if( lIdx < 64 )
		{
			u32 u0, u1, u2;
			u0 = ldsSortData[idx-3];
			u1 = ldsSortData[idx-2];
			u2 = ldsSortData[idx-1];
			AtomAdd( ldsSortData[idx], u0+u1+u2 );			
			GROUP_MEM_FENCE;

			u0 = ldsSortData[idx-12];
			u1 = ldsSortData[idx-8];
			u2 = ldsSortData[idx-4];
			AtomAdd( ldsSortData[idx], u0+u1+u2 );			
			GROUP_MEM_FENCE;

			u0 = ldsSortData[idx-48];
			u1 = ldsSortData[idx-32];
			u2 = ldsSortData[idx-16];
			AtomAdd( ldsSortData[idx], u0+u1+u2 );			
			GROUP_MEM_FENCE;

			ldsSortData[idx-1] += ldsSortData[idx-2];
			GROUP_MEM_FENCE;
		}
#else
		if( lIdx < 64 )
		{
			ldsSortData[idx] += ldsSortData[idx-1];
			GROUP_MEM_FENCE;
			ldsSortData[idx] += ldsSortData[idx-2];			
			GROUP_MEM_FENCE;

			ldsSortData[idx] += ldsSortData[idx-4];
			GROUP_MEM_FENCE;
			ldsSortData[idx] += ldsSortData[idx-8];
			GROUP_MEM_FENCE;

			ldsSortData[idx] += ldsSortData[idx-16];
			GROUP_MEM_FENCE;
			ldsSortData[idx] += ldsSortData[idx-32];
			GROUP_MEM_FENCE;
//			ldsSortData[idx] += ldsSortData[idx-64];
//			GROUP_MEM_FENCE;

			ldsSortData[idx-1] += ldsSortData[idx-2];
			GROUP_MEM_FENCE;
		}
#endif
	}

//	GROUP_LDS_BARRIER;

	totalSum = ldsSortData[wgSize*2-1];
	u32 addValue = ldsSortData[lIdx+wgSize-1];
	return addValue;
}



void sort4Bits1(inout u32 sortData[4], int startBit, int lIdx)
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
			rankPacked = localPrefixSum64VSingle( key4, lIdx, sumPacked );
		}

//		GROUP_LDS_BARRIER;

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
			prefixScanVectorEx( scannedKey );
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

//			GROUP_LDS_BARRIER;

			sortData[0] = ldsSortData[lIdx*4+0];
			sortData[1] = ldsSortData[lIdx*4+1];
			sortData[2] = ldsSortData[lIdx*4+2];
			sortData[3] = ldsSortData[lIdx*4+3];

//			GROUP_LDS_BARRIER;
		}
	}
}


groupshared u32 localHistogramToCarry[NUM_BUCKET];
groupshared u32 localHistogram[NUM_BUCKET*2];


RWStructuredBuffer<u32> gDst : register( u0 );
//StructuredBuffer<u32> gSrc : register( t0 );
StructuredBuffer<u32> rHistogram : register( t1 );


[numthreads(WG_SIZE, 1, 1)]
void SortAndScatterKernel( DEFAULT_ARGS )
{
	u32 gIdx = GET_GLOBAL_IDX;
	u32 lIdx = GET_LOCAL_IDX;
	u32 wgIdx = GET_GROUP_IDX;
	u32 wgSize = GET_GROUP_SIZE;

	const int n = m_n;
	const int nWGs = m_nWGs;
	const int startBit = m_startBit;
	const int nBlocksPerWG = m_nBlocksPerWG;

	if( lIdx < (NUM_BUCKET) )
	{
		localHistogramToCarry[lIdx] = rHistogram[lIdx*nWGs + wgIdx];
	}

	GROUP_LDS_BARRIER;

	const int blockSize = ELEMENTS_PER_WORK_ITEM*WG_SIZE;
//	for(int addr=ELEMENTS_PER_WORK_ITEM*gIdx; addr<n; addr+=stride)
	[loop]
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

		sort4Bits1(sortData, startBit, lIdx);

		u32 keys[4];
		for(int i=0; i<4; i++)
			keys[i] = (sortData[i]>>startBit) & 0xf;

		{	//	create histogram
			if( lIdx < NUM_BUCKET )
			{
				localHistogram[lIdx] = 0;
				localHistogram[NUM_BUCKET+lIdx] = 0;
			}
//			GROUP_LDS_BARRIER;

			AtomInc( localHistogram[NUM_BUCKET+keys[0]] );
			AtomInc( localHistogram[NUM_BUCKET+keys[1]] );
			AtomInc( localHistogram[NUM_BUCKET+keys[2]] );
			AtomInc( localHistogram[NUM_BUCKET+keys[3]] );
			
//			GROUP_LDS_BARRIER;
			
			uint hIdx = NUM_BUCKET+lIdx;
			if( lIdx < NUM_BUCKET )
			{
				myHistogram = localHistogram[hIdx];
			}
//			GROUP_LDS_BARRIER;

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

//			GROUP_LDS_BARRIER;
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

//		GROUP_LDS_BARRIER;

		if( lIdx < NUM_BUCKET )
		{
			localHistogramToCarry[lIdx] += myHistogram;
		}
//		GROUP_LDS_BARRIER;

	}
}


