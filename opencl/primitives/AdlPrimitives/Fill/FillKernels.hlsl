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


cbuffer CB : register( b0 )
{
	int4 m_data;
	int m_offset;
	int m_n;
	int m_padding[2];
};


RWStructuredBuffer<int> dstInt : register( u0 );

[numthreads(64, 1, 1)]
void FillIntKernel( DEFAULT_ARGS )
{
	int gIdx = GET_GLOBAL_IDX;

	if( gIdx < m_n )
	{
		dstInt[ m_offset+gIdx ] = m_data.x;
	}
}

RWStructuredBuffer<int2> dstInt2 : register( u0 );

[numthreads(64, 1, 1)]
void FillInt2Kernel( DEFAULT_ARGS )
{
	int gIdx = GET_GLOBAL_IDX;

	if( gIdx < m_n )
	{
		dstInt2[ m_offset+gIdx ] = make_int2( m_data.x, m_data.y );
	}
}

RWStructuredBuffer<int4> dstInt4 : register( u0 );

[numthreads(64, 1, 1)]
void FillInt4Kernel( DEFAULT_ARGS )
{
	int gIdx = GET_GLOBAL_IDX;

	if( gIdx < m_n )
	{
		dstInt4[ m_offset+gIdx ] = m_data;
	}
}
