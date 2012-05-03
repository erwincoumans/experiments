
#include "btcFindPairs.h"
#include "LinearMath/btAlignedObjectArray.h"
#include "BulletCollision/BroadphaseCollision/btSimpleBroadphase.h"
#include <stdio.h>

class btbBruteForceSpace : public btbAabbSpaceInterface
{
	
	btSimpleBroadphase* m_simpleBP;
	btAlignedObjectArray<void*>	m_mappedPairs;
public:
	btbBruteForceSpace()
	{
		m_simpleBP = new btSimpleBroadphase();
	}

	virtual ~btbBruteForceSpace()
	{
		delete m_simpleBP;
	}
	
	virtual btcAabbProxy btcCreateAabbProxy(btcAabbSpace bp, void* clientData, float minX,float minY,float minZ, float maxX,float maxY, float maxZ)
	{

		btVector3 aabbMin(minX,minY,minZ);
		btVector3 aabbMax(maxX,maxY,maxZ);
		void* multiSapProxy=0;
		btDispatcher* dispatcher = 0;
		int shapeType = 0;
		unsigned short int collisionFilterGroup = 1;
		unsigned short int collisionFilterMask = 1;
		
		return (btcAabbProxy) m_simpleBP->createProxy(aabbMin,aabbMax,shapeType, clientData, collisionFilterGroup, collisionFilterMask, dispatcher,multiSapProxy);
//	virtual btBroadphaseProxy*	createProxy(  const btVector3& aabbMin,  const btVector3& aabbMax,int shapeType,void* userPtr ,short int collisionFilterGroup,short int collisionFilterMask, btDispatcher* dispatcher,void* multiSapProxy);
		
	}
	virtual void btcDestroyAabbProxy(btcAabbSpace bp, btcAabbProxy proxyHandle)
	{

		btBroadphaseProxy* proxy = (btBroadphaseProxy*) proxyHandle;
		m_simpleBP->destroyProxy(proxy,0);
	}
	virtual void btcSetAabb(btcAabbSpace bp, btcAabbProxy aabbHandle, float minX,float minY,float minZ, float maxX,float maxY, float maxZ)
	{
		btBroadphaseProxy* proxy = (btBroadphaseProxy*) aabbHandle;
		btVector3 aabbMin(minX,minY,minZ);
		btVector3 aabbMax(maxX,maxY,maxZ);
		m_simpleBP->setAabb(proxy,aabbMin,aabbMax,0);
	}
	virtual int	btcFindPairs(btcAabbSpace bp)
	{
		m_simpleBP->calculateOverlappingPairs(0);
		return m_simpleBP->getOverlappingPairCache()->getNumOverlappingPairs();
	}
	virtual btbBuffer btcGetPairBuffer(btcAabbSpace bp)
	{
		//todo: typed buffer, schema etc
		return (btbBuffer)m_simpleBP->getOverlappingPairCache();
		
	}
	virtual void btcMapPairBuffer(btcAabbSpace aabbSpace, int* numPairs, unsigned char** proxyAbase, unsigned char** proxyBbase,int* proxyType, int* pairStrideInBytes)
	{
	
		*numPairs = m_simpleBP->getOverlappingPairCache()->getNumOverlappingPairs();
		if (numPairs>0)
		{
			m_simpleBP->getOverlappingPairCache()->getOverlappingPairArray()[0];

			m_mappedPairs.resize(*numPairs*2);
			
			for (int i=0;i<*numPairs;i++)
			{
				m_mappedPairs[i*2] = m_simpleBP->getOverlappingPairCache()->getOverlappingPairArray()[i].m_pProxy0->m_clientObject;
				m_mappedPairs[i*2+1] = m_simpleBP->getOverlappingPairCache()->getOverlappingPairArray()[i].m_pProxy1->m_clientObject;
				//printf("pair %d = (%p, %p)\n", *numPairs, m_simpleBP->getOverlappingPairCache()->getOverlappingPairArray()[i].m_pProxy0->m_clientObject,
				//								m_simpleBP->getOverlappingPairCache()->getOverlappingPairArray()[i].m_pProxy1->m_clientObject);
			}
			
			*proxyAbase = (unsigned char*)&(m_mappedPairs[0]);
			*proxyBbase = (unsigned char*)&(m_mappedPairs[1]);
			*proxyType = BTB_FLOAT_TYPE;
			*pairStrideInBytes = sizeof(void*)*2;
			
		}
		
	}
	virtual void btcUnmapBuffer(btcAabbSpace aabbSpace)
	{
	}
};

btcAabbSpace plCreateBruteforceSpace(int maxNumAabbs, int maxNumPairs)
{
	btbBruteForceSpace* space = new btbBruteForceSpace();
	return (btcAabbSpace) space;
}

void	plDestroySpace(btcAabbSpace bp)
{
	btbAabbSpaceInterface* space = (btbAabbSpaceInterface*)bp;
	delete space;
}

btcAabbProxy btcCreateAabbProxy(btcAabbSpace bp, void* clientData, float minX,float minY,float minZ, float maxX,float maxY, float maxZ)
{

	btbAabbSpaceInterface* space = (btbAabbSpaceInterface*)bp;
	return space->btcCreateAabbProxy(bp, clientData, minX,minY,minZ, maxX,maxY, maxZ);
}

void btcDestroyAabbProxy(btcAabbSpace bp, btcAabbProxy proxyHandle)
{
	btbAabbSpaceInterface* space = (btbAabbSpaceInterface*)bp;
	space->btcDestroyAabbProxy(bp, proxyHandle);
}

void btcSetAabb(btcAabbSpace bp, btcAabbProxy aabbHandle, float minX,float minY,float minZ, float maxX,float maxY, float maxZ)
{
	btbAabbSpaceInterface* space = (btbAabbSpaceInterface*)bp;
	space->btcSetAabb(bp,aabbHandle, minX,minY,minZ, maxX,maxY, maxZ);
}

int btcFindPairs(btcAabbSpace bp)
{
	btbAabbSpaceInterface* space = (btbAabbSpaceInterface*)bp;
	return space->btcFindPairs(bp);
	
}

btbBuffer btcGetPairBuffer(btcAabbSpace bp)
{
	//1) we don't know if the buffer is on the host or device
	//2) we don't know the layout/size of each element in the buffer
	//3) how many total elements are in the buffer?
	btbAabbSpaceInterface* space = (btbAabbSpaceInterface*)bp;
	return space->btcGetPairBuffer(bp);

}


void btcMapPairBuffer(btcAabbSpace aabbSpace, int* numPairs, unsigned char** proxyAbase, unsigned char** proxyBbase,int* proxyType, int* pairStrideInBytes)
{
	btbAabbSpaceInterface* space = (btbAabbSpaceInterface*)aabbSpace;
	space->btcMapPairBuffer(aabbSpace, numPairs, proxyAbase, proxyBbase, proxyType, pairStrideInBytes);
}
void btcUnmapBuffer(btcAabbSpace aabbSpace)
{
	btbAabbSpaceInterface* space = (btbAabbSpaceInterface*)aabbSpace;
	space->btcUnmapBuffer(aabbSpace);
}
