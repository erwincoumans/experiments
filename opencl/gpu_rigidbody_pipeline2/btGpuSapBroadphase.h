#ifndef BT_GPU_SAP_BROADPHASE_H
#define BT_GPU_SAP_BROADPHASE_H

#include "../broadphase_benchmark/btOpenCLArray.h"

struct btSapAabb
{
	union
	{
		float m_min[4];
		int m_minIndices[4];
	};
	union
	{
		float m_max[4];
		int m_signedMaxIndices[4];
		unsigned int m_unsignedMaxIndices[4];
	};
};



class btGpuSapBroadphase
{
	
	cl_context				m_context;
	cl_device_id			m_device;
	cl_command_queue	m_queue;
	
	class btRadixSort32CL* m_sorter;

	public:
	
	btOpenCLArray<btSapAabb> m_aabbs;
	btOpenCLArray<btInt2>	m_overlappingPairs;
		
	btGpuSapBroadphase(cl_context ctx,cl_device_id device, cl_command_queue  q );
	virtual ~btGpuSapBroadphase();
	
	int findOverlappingPairs();
	
};

#endif //BT_GPU_SAP_BROADPHASE_H