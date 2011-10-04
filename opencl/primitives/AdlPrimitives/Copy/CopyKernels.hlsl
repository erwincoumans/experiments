/*
		2011 Takahiro Harada
*/

typedef uint u32;

#define GET_GROUP_IDX groupIdx.x
#define GET_LOCAL_IDX localIdx.x
#define GET_GLOBAL_IDX globalIdx.x
#define GROUP_LDS_BARRIER GroupMemoryBarrierWithGroupSync()
#define GROUP_MEM_FENCE
#define DEFAULT_ARGS uint3 globalIdx : SV_DispatchThreadID, uint3 localIdx : SV_GroupThreadID, uint3 groupIdx : SV_GroupID
#define AtomInc(x) InterlockedAdd(x, 1)
#define AtomInc1(x, out) InterlockedAdd(x, 1, out)

#define make_uint4 uint4
#define make_uint2 uint2
#define make_int2 int2

#define WG_SIZE 64

#define GET_GROUP_SIZE WG_SIZE



cbuffer CB : register( b0 )
{
	int m_n;
	int m_padding[3];
};

RWStructuredBuffer<float4> dst : register( u0 );
StructuredBuffer<float4> src : register( t0 );

[numthreads(WG_SIZE, 1, 1)]
void Copy1F4Kernel( DEFAULT_ARGS )
{
	int gIdx = GET_GLOBAL_IDX;

	if( gIdx < m_n )
	{
		float4 a0 = src[gIdx];

		dst[ gIdx ] = a0;
	}
}

[numthreads(WG_SIZE, 1, 1)]
void Copy2F4Kernel( DEFAULT_ARGS )
{
	int gIdx = GET_GLOBAL_IDX;

	if( 2*gIdx <= m_n )
	{
		float4 a0 = src[gIdx*2+0];
		float4 a1 = src[gIdx*2+1];

		dst[ gIdx*2+0 ] = a0;
		dst[ gIdx*2+1 ] = a1;
	}
}

[numthreads(WG_SIZE, 1, 1)]
void Copy4F4Kernel( DEFAULT_ARGS )
{
	int gIdx = GET_GLOBAL_IDX;

	if( 4*gIdx <= m_n )
	{
		int idx0 = gIdx*4+0;
		int idx1 = gIdx*4+1;
		int idx2 = gIdx*4+2;
		int idx3 = gIdx*4+3;

		float4 a0 = src[idx0];
		float4 a1 = src[idx1];
		float4 a2 = src[idx2];
		float4 a3 = src[idx3];

		dst[ idx0 ] = a0;
		dst[ idx1 ] = a1;
		dst[ idx2 ] = a2;
		dst[ idx3 ] = a3;
	}
}

RWStructuredBuffer<float> dstF1 : register( u0 );
StructuredBuffer<float> srcF1 : register( t0 );

[numthreads(WG_SIZE, 1, 1)]
void CopyF1Kernel( DEFAULT_ARGS )
{
	int gIdx = GET_GLOBAL_IDX;

	if( gIdx < m_n )
	{
		float a0 = srcF1[gIdx];

		dstF1[ gIdx ] = a0;
	}

}

RWStructuredBuffer<float2> dstF2 : register( u0 );
StructuredBuffer<float2> srcF2 : register( t0 );

[numthreads(WG_SIZE, 1, 1)]
void CopyF2Kernel( DEFAULT_ARGS )
{
	int gIdx = GET_GLOBAL_IDX;

	if( gIdx < m_n )
	{
		float2 a0 = srcF2[gIdx];

		dstF2[ gIdx ] = a0;
	}
}
