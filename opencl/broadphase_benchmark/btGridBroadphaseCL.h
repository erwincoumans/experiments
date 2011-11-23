#ifndef GRID_BROADPHASE_CL_H
#define GRID_BROADPHASE_CL_H

#include "../3dGridBroadphase/Shared/bt3dGridBroadphaseOCL.h"

#include "Adl/Adl.h"
#include "Adl/AdlKernel.h"


struct MyAabbConstData 
{
	int bla;
	int numElem;
};

struct MyUint2
{
	int x;
	int y;
};

class btGridBroadphaseCl : public bt3dGridBroadphaseOCL
{
protected:

	adl::Kernel*			m_computeAabbKernel;
	adl::Kernel*			m_computePairChanges;
	adl::Kernel*			m_squeezePairCaches;


	adl::Buffer<MyAabbConstData>*	m_aabbConstBuffer;


	public:

		cl_mem					m_dPairsChangedXY;
		MyUint2*				m_hPairsChangedXY;
		
		btGridBroadphaseCl(	btOverlappingPairCache* overlappingPairCache,
							const btVector3& cellSize, 
							int gridSizeX, int gridSizeY, int gridSizeZ, 
							int maxSmallProxies, int maxLargeProxies, int maxPairsPerSmallProxy,
							btScalar maxSmallProxySize,
							int maxSmallProxiesPerCell = 8,
							cl_context context = NULL,
							cl_device_id device = NULL,
							cl_command_queue queue = NULL);
		
		virtual void prepareAABB(float* positions, int numObjects);
		virtual void calcHashAABB();

		void calculateOverlappingPairs(float* positions, int numObjects);
		
		virtual ~btGridBroadphaseCl();							
	
};

#endif //GRID_BROADPHASE_CL_H

