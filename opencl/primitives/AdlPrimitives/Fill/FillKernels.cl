/*
		2011 Takahiro Harada
*/

#pragma OPENCL EXTENSION cl_amd_printf : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable

typedef unsigned int u32;
#define GET_GROUP_IDX get_group_id(0)
#define GET_LOCAL_IDX get_local_id(0)
#define GET_GLOBAL_IDX get_global_id(0)
#define GET_GROUP_SIZE get_local_size(0)
#define GROUP_LDS_BARRIER barrier(CLK_LOCAL_MEM_FENCE)
#define GROUP_MEM_FENCE mem_fence(CLK_LOCAL_MEM_FENCE)
#define AtomInc(x) atom_inc(&(x))
#define AtomInc1(x, out) out = atom_inc(&(x))

#define make_uint4 (uint4)
#define make_uint2 (uint2)
#define make_int2 (int2)

typedef struct
{
	int4 m_data;
	int m_offset;
	int m_n;
	int m_padding[2];
} ConstBuffer;


__kernel
__attribute__((reqd_work_group_size(64,1,1)))
void FillIntKernel(__global int* dstInt, 
					ConstBuffer cb)
{
	int gIdx = GET_GLOBAL_IDX;

	if( gIdx < cb.m_n )
	{
		dstInt[ cb.m_offset+gIdx ] = cb.m_data.x;
	}
}

__kernel
__attribute__((reqd_work_group_size(64,1,1)))
void FillInt2Kernel(__global int2* dstInt2, 
					ConstBuffer cb)
{
	int gIdx = GET_GLOBAL_IDX;

	if( gIdx < cb.m_n )
	{
		dstInt2[ cb.m_offset+gIdx ] = make_int2( cb.m_data.x, cb.m_data.y );
	}
}

__kernel
__attribute__((reqd_work_group_size(64,1,1)))
void FillInt4Kernel(__global int4* dstInt4, 
					ConstBuffer cb)
{
	int gIdx = GET_GLOBAL_IDX;

	if( gIdx < cb.m_n )
	{
		dstInt4[ cb.m_offset+gIdx ] = cb.m_data;
	}
}

