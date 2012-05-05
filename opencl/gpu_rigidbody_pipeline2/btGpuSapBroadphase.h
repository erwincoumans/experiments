#ifndef BT_GPU_SAP_BROADPHASE_H
#define BT_GPU_SAP_BROADPHASE_H

#include "../broadphase_benchmark/btOpenCLArray.h"
#include "../broadphase_benchmark/btFillCL.h" //btInt2
class btVector3;
#include "../broadphase_benchmark/btRadixSort32CL.h"

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
	cl_command_queue		m_queue;
	cl_kernel				m_flipFloatKernel;
	cl_kernel				m_scatterKernel ;
	cl_kernel				m_sapKernel;

	class btRadixSort32CL* m_sorter;

	public:
	
	btOpenCLArray<btSapAabb>	m_aabbs;
	btOpenCLArray<btInt2>		m_overlappingPairs;

	//temporary gpu work memory
	btOpenCLArray<btSortData>	m_gpuSortData;
	btOpenCLArray<btSapAabb>	m_gpuSortedAabbs;


	btGpuSapBroadphase(cl_context ctx,cl_device_id device, cl_command_queue  q );
	virtual ~btGpuSapBroadphase();
	
	void  calculateOverlappingPairs();

	void createProxy(const btVector3& aabbMin,  const btVector3& aabbMax,int shapeType,
				void* userPtr ,short int collisionFilterGroup,short int collisionFilterMask);

	cl_mem	getAabbBuffer();
	int	getNumOverlap();
	cl_mem	getOverlappingPairBuffer();
};

#endif //BT_GPU_SAP_BROADPHASE_H