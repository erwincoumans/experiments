/*
		2011 Takahiro Harada
*/

typedef uint u32;

#define GET_GROUP_IDX groupIdx.x
#define GET_LOCAL_IDX localIdx.x
#define GET_GLOBAL_IDX globalIdx.x
#define GROUP_LDS_BARRIER GroupMemoryBarrierWithGroupSync()
#define DEFAULT_ARGS uint3 globalIdx : SV_DispatchThreadID, uint3 localIdx : SV_GroupThreadID, uint3 groupIdx : SV_GroupID
#define AtomInc(x) InterlockedAdd(x, 1)
#define AtomInc1(x, out) InterlockedAdd(x, 1, out)



typedef struct
{
	u32 m_key; 
	u32 m_value;
}SortData;



cbuffer SortCB : register( b0 )
{
	u32 m_nSrc;
	u32 m_nDst;
	u32 m_padding[2];
};


StructuredBuffer<SortData> src : register( t0 );
RWStructuredBuffer<u32> dst : register( u0 );


[numthreads(64, 1, 1)]
void SearchSortDataLowerKernel( DEFAULT_ARGS )
{
	int gIdx = GET_GLOBAL_IDX;
	u32 nSrc = m_nSrc;
	u32 nDst = m_nDst;

	if( gIdx < nSrc )
	{
		SortData iData;
		SortData jData;
		if( gIdx==0 ) iData.m_key = iData.m_value = (u32)-1;
		else iData = src[gIdx-1];

		if( gIdx==nSrc ) jData.m_key = jData.m_value = nDst;
		else jData = src[gIdx];

		if( iData.m_key != jData.m_key )
		{
//			for(u32 k=iData.m_key+1; k<=min(jData.m_key, nDst-1); k++)
			u32 k = jData.m_key;
			{
				dst[k] = gIdx;
			}
		}
	}
}

[numthreads(64, 1, 1)]
void SearchSortDataUpperKernel( DEFAULT_ARGS )
{
	int gIdx = GET_GLOBAL_IDX;
	u32 nSrc = m_nSrc;
	u32 nDst = m_nDst;

	if( gIdx < nSrc+1 )
	{
		SortData iData;
		SortData jData;
		if( gIdx==0 ) iData.m_key = iData.m_value = 0;
		else iData = src[gIdx-1];

		if( gIdx==nSrc ) jData.m_key = jData.m_value = nDst;
		else jData = src[gIdx];

		if( iData.m_key != jData.m_key )
		{
//			for(u32 k=iData.m_key; k<min(jData.m_key, nDst); k++)
			u32 k = iData.m_key;
			{
				dst[k] = gIdx;
			}
		}
	}
}

