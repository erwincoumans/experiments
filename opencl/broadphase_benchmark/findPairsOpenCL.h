
#ifndef FIND_PAIRS_H
#define FIND_PAIRS_H

#include "../basic_initialize/btOpenCLInclude.h"

struct btKernelInfo
{
	int			m_Id;
	cl_kernel	m_kernel;
	char*		m_name;
	int			m_workgroupSize;
};



struct btFindPairsIO
{
	int				m_numObjects;

	cl_mem			m_clObjectsBuffer; //for memory layout details see main.cpp (todo, make it flexible)
	int				m_positionOffset;//offset in m_clObjectsBuffer where position array starts

	cl_command_queue			m_cqCommandQue;
	cl_kernel		m_broadphaseGridKernel;
	cl_kernel	m_broadphaseColorKernel;
	cl_kernel	m_broadphaseBruteForceKernel;

	cl_context		m_mainContext;
	cl_device_id	m_device;

	cl_kernel		m_calcHashAabbKernel;
	cl_kernel		m_clearCellStartKernel;
	cl_kernel		m_findCellStartKernel;
	cl_kernel		m_findOverlappingPairsKernel;
	cl_kernel		m_computePairChangeKernel;
	cl_kernel		m_squeezePairBuffKernel;


	cl_mem m_dPairsChangedXY;
	int m_numOverlap;

	cl_mem					m_dBpParams;
	cl_mem					m_dBodiesHash;
	cl_mem					m_dCellStart;
	cl_mem					m_dPairBuff; 
	cl_mem					m_dPairBuffStartCurr;
	cl_mem					m_dAABB;
	cl_mem					m_dPairScan;
	cl_mem					m_dPairOut;
};


void initFindPairs(btFindPairsIO& fpio,cl_context cxMainContext, cl_device_id device, cl_command_queue commandQueue, int maxHandles,int maxPairsPerBody = 16);

void findPairsOpenCL(btFindPairsIO& fpio);

void	drawPairsOpenCL(btFindPairsIO&	fpio);


void releaseFindPairs(btFindPairsIO& fpio);

#endif //FIND_PAIRS_H
