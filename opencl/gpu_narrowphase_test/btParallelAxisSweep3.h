//Bullet Continuous Collision Detection and Physics Library
//Copyright (c) 2003-2006 Erwin Coumans  http://continuousphysics.com/Bullet/

//
// btAxisSweep3.h
//
// Copyright (c) 2006 Simon Hobbs
//
// This software is provided 'as-is', without any express or implied warranty. In no event will the authors be held liable for any damages arising from the use of this software.
//
// Permission is granted to anyone to use this software for any purpose, including commercial applications, and to alter it and redistribute it freely, subject to the following restrictions:
//
// 1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
//
// 2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
//
// 3. This notice may not be removed or altered from any source distribution.

#ifndef BT_PARALLEL_AXIS_SWEEP_3_H
#define BT_PARALLEL_AXIS_SWEEP_3_H

#include "LinearMath/btVector3.h"
#include "BulletCollision/BroadphaseCollision/btOverlappingPairCache.h"
#include "BulletCollision/BroadphaseCollision/btBroadphaseInterface.h"
#include "BulletCollision/BroadphaseCollision/btBroadphaseProxy.h"
#include "BulletCollision/BroadphaseCollision/btOverlappingPairCallback.h"
#include "BulletCollision/BroadphaseCollision/btDbvtBroadphase.h"
#include "LinearMath/btQuickProf.h"
//#define DEBUG_BROADPHASE 1
#define USE_OVERLAP_TEST_ON_REMOVES 1
static bool gVerbose = false;
/// The internal templace class btParallelAxisSweep3Internal implements the sweep and prune broadphase.
/// It uses quantized integers to represent the begin and end points for each of the 3 axis.
/// Dont use this class directly, use btAxisSweep3 or bt32BitAxisSweep3 instead.
template <typename BP_FP_INT_TYPE>
class btParallelAxisSweep3Internal : public btBroadphaseInterface
{
protected:

	BP_FP_INT_TYPE	m_bpHandleMask;
	BP_FP_INT_TYPE	m_handleSentinel;

public:
	
 BT_DECLARE_ALIGNED_ALLOCATOR();

	class Edge
	{
	public:
		BP_FP_INT_TYPE m_pos;			// low bit is min/max
		BP_FP_INT_TYPE m_handle;

		BP_FP_INT_TYPE IsMax() const {return static_cast<BP_FP_INT_TYPE>(m_pos & 1);}
	};

public:
	class	Handle : public btBroadphaseProxy
	{
	public:
	BT_DECLARE_ALIGNED_ALLOCATOR();
	
		// indexes into the edge arrays
		BP_FP_INT_TYPE m_minEdges[3], m_maxEdges[3];		// 6 * 2 = 12
//		BP_FP_INT_TYPE m_uniqueId;
		btBroadphaseProxy*	m_dbvtProxy;//for faster raycast
		//void* m_pOwner; this is now in btBroadphaseProxy.m_clientObject
	
		SIMD_FORCE_INLINE void SetNextFree(BP_FP_INT_TYPE next) {m_minEdges[0] = next;}
		SIMD_FORCE_INLINE BP_FP_INT_TYPE GetNextFree() const {return m_minEdges[0];}
	};		// 24 bytes + 24 for Edge structures = 44 bytes total per entry

	
protected:
	btVector3 m_worldAabbMin;						// overall system bounds
	btVector3 m_worldAabbMax;						// overall system bounds

	btVector3 m_quantize;						// scaling factor for quantization

	BP_FP_INT_TYPE m_numHandles;						// number of active handles
	BP_FP_INT_TYPE m_maxHandles;						// max number of handles
	Handle* m_pHandles;						// handles pool
	
	BP_FP_INT_TYPE m_firstFreeHandle;		// free handles list

	Edge* m_pEdges[3];						// edge arrays for the 3 axes (each array has m_maxHandles * 2 + 2 sentinel entries)
	void* m_pEdgesRawPtr[3];

	//////

	btAlignedObjectArray<Handle> m_handles[2];
	btAlignedObjectArray<Edge> m_edges[2][3];

	int	m_frontBufferIndex;

	btOverlappingPairCache* m_pairCache;
	btOverlappingPairCache* m_pairCache2;

	///btOverlappingPairCallback is an additional optional user callback for adding/removing overlapping pairs, similar interface to btOverlappingPairCache.
	btOverlappingPairCallback* m_userPairCallback;
	
	bool	m_ownsPairCache;

	int	m_invalidPair;

	///additional dynamic aabb structure, used to accelerate ray cast queries.
	///can be disabled using a optional argument in the constructor
	btDbvtBroadphase*	m_raycastAccelerator;
	btOverlappingPairCache*	m_nullPairCache;


	// allocation/deallocation
	BP_FP_INT_TYPE allocHandle();
	void freeHandle(BP_FP_INT_TYPE handle);
	

	bool testOverlap2D(const Handle* pHandleA, const Handle* pHandleB,int axis0,int axis1);

#ifdef DEBUG_BROADPHASE
	void debugPrintAxis(int axis,bool checkCardinality=true);
#endif //DEBUG_BROADPHASE

	//Overlap* AddOverlap(BP_FP_INT_TYPE handleA, BP_FP_INT_TYPE handleB);
	//void RemoveOverlap(BP_FP_INT_TYPE handleA, BP_FP_INT_TYPE handleB);

	

	void sortMinDown(int axis, BP_FP_INT_TYPE edge, btDispatcher* dispatcher, bool updateOverlaps );
	void sortMinUp(int axis, BP_FP_INT_TYPE edge, btDispatcher* dispatcher, bool updateOverlaps );
	void sortMaxDown(int axis, BP_FP_INT_TYPE edge, btDispatcher* dispatcher, bool updateOverlaps );
	void sortMaxUp(int axis, BP_FP_INT_TYPE edge, btDispatcher* dispatcher, bool updateOverlaps );

public:

	btParallelAxisSweep3Internal(const btVector3& worldAabbMin,const btVector3& worldAabbMax, BP_FP_INT_TYPE handleMask, BP_FP_INT_TYPE handleSentinel, BP_FP_INT_TYPE maxHandles = 16384, btOverlappingPairCache* pairCache=0,bool disableRaycastAccelerator = false);

	virtual	~btParallelAxisSweep3Internal();

	BP_FP_INT_TYPE getNumHandles() const
	{
		return m_numHandles;
	}

	virtual void	calculateOverlappingPairs(btDispatcher* dispatcher);
	
	BP_FP_INT_TYPE addHandle(const btVector3& aabbMin,const btVector3& aabbMax, void* pOwner,short int collisionFilterGroup,short int collisionFilterMask,btDispatcher* dispatcher,void* multiSapProxy);
	void removeHandle(BP_FP_INT_TYPE handle,btDispatcher* dispatcher);
	void updateHandle(BP_FP_INT_TYPE handle, const btVector3& aabbMin,const btVector3& aabbMax,btDispatcher* dispatcher);
	SIMD_FORCE_INLINE Handle* getHandle(BP_FP_INT_TYPE index) const {return m_pHandles + index;}

	virtual void resetPool(btDispatcher* dispatcher);

	void	processAllOverlappingPairs(btOverlapCallback* callback);

	//Broadphase Interface
	virtual btBroadphaseProxy*	createProxy(  const btVector3& aabbMin,  const btVector3& aabbMax,int shapeType,void* userPtr ,short int collisionFilterGroup,short int collisionFilterMask,btDispatcher* dispatcher,void* multiSapProxy);
	virtual void	destroyProxy(btBroadphaseProxy* proxy,btDispatcher* dispatcher);
	virtual void	setAabb(btBroadphaseProxy* proxy,const btVector3& aabbMin,const btVector3& aabbMax,btDispatcher* dispatcher);
	virtual void  getAabb(btBroadphaseProxy* proxy,btVector3& aabbMin, btVector3& aabbMax ) const;
	
	virtual void	rayTest(const btVector3& rayFrom,const btVector3& rayTo, btBroadphaseRayCallback& rayCallback, const btVector3& aabbMin=btVector3(0,0,0), const btVector3& aabbMax = btVector3(0,0,0));
	virtual void	aabbTest(const btVector3& aabbMin, const btVector3& aabbMax, btBroadphaseAabbCallback& callback);

	
	void quantize(BP_FP_INT_TYPE* out, const btVector3& point, int isMax) const;
	///unQuantize should be conservative: aabbMin/aabbMax should be larger then 'getAabb' result
	void unQuantize(btBroadphaseProxy* proxy,btVector3& aabbMin, btVector3& aabbMax ) const;
	
	bool	testAabbOverlap(btBroadphaseProxy* proxy0,btBroadphaseProxy* proxy1);
	bool slowTestAabbOverlap(btBroadphaseProxy* proxy0,btBroadphaseProxy* proxy1);

	btOverlappingPairCache*	getOverlappingPairCache()
	{
		return m_pairCache;
	}
	const btOverlappingPairCache*	getOverlappingPairCache() const
	{
		return m_pairCache;
	}

	void	setOverlappingPairUserCallback(btOverlappingPairCallback* pairCallback)
	{
		m_userPairCallback = pairCallback;
	}
	const btOverlappingPairCallback*	getOverlappingPairUserCallback() const
	{
		return m_userPairCallback;
	}

	///getAabb returns the axis aligned bounding box in the 'global' coordinate frame
	///will add some transform later
	virtual void getBroadphaseAabb(btVector3& aabbMin,btVector3& aabbMax) const
	{
		aabbMin = m_worldAabbMin;
		aabbMax = m_worldAabbMax;
	}

	virtual void	printStats()
	{
/*		printf("btAxisSweep3.h\n");
		printf("numHandles = %d, maxHandles = %d\n",m_numHandles,m_maxHandles);
		printf("aabbMin=%f,%f,%f,aabbMax=%f,%f,%f\n",m_worldAabbMin.getX(),m_worldAabbMin.getY(),m_worldAabbMin.getZ(),
			m_worldAabbMax.getX(),m_worldAabbMax.getY(),m_worldAabbMax.getZ());
			*/

	}

};

////////////////////////////////////////////////////////////////////




#ifdef DEBUG_BROADPHASE
#include <stdio.h>

template <typename BP_FP_INT_TYPE>
void btAxisSweep3<BP_FP_INT_TYPE>::debugPrintAxis(int axis, bool checkCardinality)
{
	int numEdges = m_pHandles[0].m_maxEdges[axis];
	printf("SAP Axis %d, numEdges=%d\n",axis,numEdges);

	int i;
	for (i=0;i<numEdges+1;i++)
	{
		Edge* pEdge = m_pEdges[axis] + i;
		Handle* pHandlePrev = getHandle(pEdge->m_handle);
		int handleIndex = pEdge->IsMax()? pHandlePrev->m_maxEdges[axis] : pHandlePrev->m_minEdges[axis];
		char beginOrEnd;
		beginOrEnd=pEdge->IsMax()?'E':'B';
		printf("	[%c,h=%d,p=%x,i=%d]\n",beginOrEnd,pEdge->m_handle,pEdge->m_pos,handleIndex);
	}

	if (checkCardinality)
		btAssert(numEdges == m_numHandles*2+1);
}
#endif //DEBUG_BROADPHASE

template <typename BP_FP_INT_TYPE>
btBroadphaseProxy*	btParallelAxisSweep3Internal<BP_FP_INT_TYPE>::createProxy(  const btVector3& aabbMin,  const btVector3& aabbMax,int shapeType,void* userPtr,short int collisionFilterGroup,short int collisionFilterMask,btDispatcher* dispatcher,void* multiSapProxy)
{
		(void)shapeType;
		BP_FP_INT_TYPE handleId = addHandle(aabbMin,aabbMax, userPtr,collisionFilterGroup,collisionFilterMask,dispatcher,multiSapProxy);
		
		Handle* handle = getHandle(handleId);
		
		if (m_raycastAccelerator)
		{
			btBroadphaseProxy* rayProxy = m_raycastAccelerator->createProxy(aabbMin,aabbMax,shapeType,userPtr,collisionFilterGroup,collisionFilterMask,dispatcher,0);
			handle->m_dbvtProxy = rayProxy;
		}
		return handle;
}



template <typename BP_FP_INT_TYPE>
void	btParallelAxisSweep3Internal<BP_FP_INT_TYPE>::destroyProxy(btBroadphaseProxy* proxy,btDispatcher* dispatcher)
{
	Handle* handle = static_cast<Handle*>(proxy);
	if (m_raycastAccelerator)
		m_raycastAccelerator->destroyProxy(handle->m_dbvtProxy,dispatcher);
	removeHandle(static_cast<BP_FP_INT_TYPE>(handle->m_uniqueId), dispatcher);
}

template <typename BP_FP_INT_TYPE>
void	btParallelAxisSweep3Internal<BP_FP_INT_TYPE>::setAabb(btBroadphaseProxy* proxy,const btVector3& aabbMin,const btVector3& aabbMax,btDispatcher* dispatcher)
{
	Handle* handle = static_cast<Handle*>(proxy);
	handle->m_aabbMin = aabbMin;
	handle->m_aabbMax = aabbMax;
	updateHandle(static_cast<BP_FP_INT_TYPE>(handle->m_uniqueId), aabbMin, aabbMax,dispatcher);
	if (m_raycastAccelerator)
		m_raycastAccelerator->setAabb(handle->m_dbvtProxy,aabbMin,aabbMax,dispatcher);

}

template <typename BP_FP_INT_TYPE>
void	btParallelAxisSweep3Internal<BP_FP_INT_TYPE>::rayTest(const btVector3& rayFrom,const btVector3& rayTo, btBroadphaseRayCallback& rayCallback,const btVector3& aabbMin,const btVector3& aabbMax)
{
	if (m_raycastAccelerator)
	{
		m_raycastAccelerator->rayTest(rayFrom,rayTo,rayCallback,aabbMin,aabbMax);
	} else
	{
		//choose axis?
		BP_FP_INT_TYPE axis = 0;
		//for each proxy
		for (BP_FP_INT_TYPE i=1;i<m_numHandles*2+1;i++)
		{
			if (m_pEdges[axis][i].IsMax())
			{
				rayCallback.process(getHandle(m_pEdges[axis][i].m_handle));
			}
		}
	}
}

template <typename BP_FP_INT_TYPE>
void	btParallelAxisSweep3Internal<BP_FP_INT_TYPE>::aabbTest(const btVector3& aabbMin, const btVector3& aabbMax, btBroadphaseAabbCallback& callback)
{
	if (m_raycastAccelerator)
	{
		m_raycastAccelerator->aabbTest(aabbMin,aabbMax,callback);
	} else
	{
		//choose axis?
		BP_FP_INT_TYPE axis = 0;
		//for each proxy
		for (BP_FP_INT_TYPE i=1;i<m_numHandles*2+1;i++)
		{
			if (m_pEdges[axis][i].IsMax())
			{
				Handle* handle = getHandle(m_pEdges[axis][i].m_handle);
				if (TestAabbAgainstAabb2(aabbMin,aabbMax,handle->m_aabbMin,handle->m_aabbMax))
				{
					callback.process(handle);
				}
			}
		}
	}
}



template <typename BP_FP_INT_TYPE>
void btParallelAxisSweep3Internal<BP_FP_INT_TYPE>::getAabb(btBroadphaseProxy* proxy,btVector3& aabbMin, btVector3& aabbMax ) const
{
	Handle* pHandle = static_cast<Handle*>(proxy);
	aabbMin = pHandle->m_aabbMin;
	aabbMax = pHandle->m_aabbMax;
}


template <typename BP_FP_INT_TYPE>
void btParallelAxisSweep3Internal<BP_FP_INT_TYPE>::unQuantize(btBroadphaseProxy* proxy,btVector3& aabbMin, btVector3& aabbMax ) const
{
	Handle* pHandle = static_cast<Handle*>(proxy);

	unsigned short vecInMin[3];
	unsigned short vecInMax[3];

	vecInMin[0] = m_pEdges[0][pHandle->m_minEdges[0]].m_pos ;
	vecInMax[0] = m_pEdges[0][pHandle->m_maxEdges[0]].m_pos +1 ;
	vecInMin[1] = m_pEdges[1][pHandle->m_minEdges[1]].m_pos ;
	vecInMax[1] = m_pEdges[1][pHandle->m_maxEdges[1]].m_pos +1 ;
	vecInMin[2] = m_pEdges[2][pHandle->m_minEdges[2]].m_pos ;
	vecInMax[2] = m_pEdges[2][pHandle->m_maxEdges[2]].m_pos +1 ;
	
	aabbMin.setValue((btScalar)(vecInMin[0]) / (m_quantize.getX()),(btScalar)(vecInMin[1]) / (m_quantize.getY()),(btScalar)(vecInMin[2]) / (m_quantize.getZ()));
	aabbMin += m_worldAabbMin;
	
	aabbMax.setValue((btScalar)(vecInMax[0]) / (m_quantize.getX()),(btScalar)(vecInMax[1]) / (m_quantize.getY()),(btScalar)(vecInMax[2]) / (m_quantize.getZ()));
	aabbMax += m_worldAabbMin;
}




template <typename BP_FP_INT_TYPE>
btParallelAxisSweep3Internal<BP_FP_INT_TYPE>::btParallelAxisSweep3Internal(const btVector3& worldAabbMin,const btVector3& worldAabbMax, BP_FP_INT_TYPE handleMask, BP_FP_INT_TYPE handleSentinel,BP_FP_INT_TYPE userMaxHandles, btOverlappingPairCache* pairCache , bool disableRaycastAccelerator)
:m_bpHandleMask(handleMask),
m_handleSentinel(handleSentinel),
m_pairCache(pairCache),
m_pairCache2(0),
m_userPairCallback(0),
m_ownsPairCache(false),
m_invalidPair(0),
m_raycastAccelerator(0)
{
	BP_FP_INT_TYPE maxHandles = static_cast<BP_FP_INT_TYPE>(userMaxHandles+1);//need to add one sentinel handle

	if (!m_pairCache)
	{
		void* ptr = btAlignedAlloc(sizeof(btHashedOverlappingPairCache),16);
		m_pairCache = new(ptr) btHashedOverlappingPairCache();
		m_ownsPairCache = true;
	}

	if (!m_pairCache2)
	{
		void* ptr = btAlignedAlloc(sizeof(btHashedOverlappingPairCache),16);
		m_pairCache2 = new(ptr) btHashedOverlappingPairCache();
	}


	if (!disableRaycastAccelerator)
	{
		m_nullPairCache = new (btAlignedAlloc(sizeof(btNullPairCache),16)) btNullPairCache();
		m_raycastAccelerator = new (btAlignedAlloc(sizeof(btDbvtBroadphase),16)) btDbvtBroadphase(m_nullPairCache);//m_pairCache);
		m_raycastAccelerator->m_deferedcollide = true;//don't add/remove pairs
	}

	//btAssert(bounds.HasVolume());

	// init bounds
	m_worldAabbMin = worldAabbMin;
	m_worldAabbMax = worldAabbMax;

	btVector3 aabbSize = m_worldAabbMax - m_worldAabbMin;

	BP_FP_INT_TYPE	maxInt = m_handleSentinel;

	m_quantize = btVector3(btScalar(maxInt),btScalar(maxInt),btScalar(maxInt)) / aabbSize;

	// allocate handles buffer, using btAlignedAlloc, and put all handles on free list
	m_pHandles = new Handle[maxHandles];
	
	m_maxHandles = maxHandles;
	m_numHandles = 0;

	m_handles[0].reserve(8192);
	m_handles[1].reserve(8192);

	for (int ax=0;ax<3;ax++)
	{
		m_edges[0][ax].reserve(8192*2);
		m_edges[1][ax].reserve(8192*2);
	}
	m_frontBufferIndex = 0;

	// handle 0 is reserved as the null index, and is also used as the sentinel
	m_firstFreeHandle = 1;
	{
		for (BP_FP_INT_TYPE i = m_firstFreeHandle; i < maxHandles; i++)
			m_pHandles[i].SetNextFree(static_cast<BP_FP_INT_TYPE>(i + 1));
		m_pHandles[maxHandles - 1].SetNextFree(0);
	}

	{
		// allocate edge buffers
		for (int i = 0; i < 3; i++)
		{
			m_pEdgesRawPtr[i] = btAlignedAlloc(sizeof(Edge)*maxHandles*2,16);
			m_pEdges[i] = new(m_pEdgesRawPtr[i]) Edge[maxHandles * 2];
		}
	}
	//removed overlap management

	// make boundary sentinels
	
	m_pHandles[0].m_clientObject = 0;

	for (int axis = 0; axis < 3; axis++)
	{
		m_pHandles[0].m_minEdges[axis] = 0;
		m_pHandles[0].m_maxEdges[axis] = 1;

		m_pEdges[axis][0].m_pos = 0;
		m_pEdges[axis][0].m_handle = 0;
		m_pEdges[axis][1].m_pos = m_handleSentinel;
		m_pEdges[axis][1].m_handle = 0;
#ifdef DEBUG_BROADPHASE
		debugPrintAxis(axis);
#endif //DEBUG_BROADPHASE

	}

}

template <typename BP_FP_INT_TYPE>
btParallelAxisSweep3Internal<BP_FP_INT_TYPE>::~btParallelAxisSweep3Internal()
{
	if (m_raycastAccelerator)
	{
		m_nullPairCache->~btOverlappingPairCache();
		btAlignedFree(m_nullPairCache);
		m_raycastAccelerator->~btDbvtBroadphase();
		btAlignedFree (m_raycastAccelerator);
	}

	for (int i = 2; i >= 0; i--)
	{
		btAlignedFree(m_pEdgesRawPtr[i]);
	}
	delete [] m_pHandles;

	if (m_ownsPairCache)
	{
		m_pairCache->~btOverlappingPairCache();
		btAlignedFree(m_pairCache);
	}
}

template <typename BP_FP_INT_TYPE>
void btParallelAxisSweep3Internal<BP_FP_INT_TYPE>::quantize(BP_FP_INT_TYPE* out, const btVector3& point, int isMax) const
{
#ifdef OLD_CLAMPING_METHOD
	///problem with this clamping method is that the floating point during quantization might still go outside the range [(0|isMax) .. (m_handleSentinel&m_bpHandleMask]|isMax]
	///see http://code.google.com/p/bullet/issues/detail?id=87
	btVector3 clampedPoint(point);
	clampedPoint.setMax(m_worldAabbMin);
	clampedPoint.setMin(m_worldAabbMax);
	btVector3 v = (clampedPoint - m_worldAabbMin) * m_quantize;
	out[0] = (BP_FP_INT_TYPE)(((BP_FP_INT_TYPE)v.getX() & m_bpHandleMask) | isMax);
	out[1] = (BP_FP_INT_TYPE)(((BP_FP_INT_TYPE)v.getY() & m_bpHandleMask) | isMax);
	out[2] = (BP_FP_INT_TYPE)(((BP_FP_INT_TYPE)v.getZ() & m_bpHandleMask) | isMax);
#else
	btVector3 v = (point - m_worldAabbMin) * m_quantize;
	out[0]=(v[0]<=0)?(BP_FP_INT_TYPE)isMax:(v[0]>=m_handleSentinel)?(BP_FP_INT_TYPE)((m_handleSentinel&m_bpHandleMask)|isMax):(BP_FP_INT_TYPE)(((BP_FP_INT_TYPE)v[0]&m_bpHandleMask)|isMax);
	out[1]=(v[1]<=0)?(BP_FP_INT_TYPE)isMax:(v[1]>=m_handleSentinel)?(BP_FP_INT_TYPE)((m_handleSentinel&m_bpHandleMask)|isMax):(BP_FP_INT_TYPE)(((BP_FP_INT_TYPE)v[1]&m_bpHandleMask)|isMax);
	out[2]=(v[2]<=0)?(BP_FP_INT_TYPE)isMax:(v[2]>=m_handleSentinel)?(BP_FP_INT_TYPE)((m_handleSentinel&m_bpHandleMask)|isMax):(BP_FP_INT_TYPE)(((BP_FP_INT_TYPE)v[2]&m_bpHandleMask)|isMax);
#endif //OLD_CLAMPING_METHOD
}


template <typename BP_FP_INT_TYPE>
BP_FP_INT_TYPE btParallelAxisSweep3Internal<BP_FP_INT_TYPE>::allocHandle()
{
	btAssert(m_firstFreeHandle);

	BP_FP_INT_TYPE handle = m_firstFreeHandle;
	m_firstFreeHandle = getHandle(handle)->GetNextFree();
	m_numHandles++;

	return handle;
}

template <typename BP_FP_INT_TYPE>
void btParallelAxisSweep3Internal<BP_FP_INT_TYPE>::freeHandle(BP_FP_INT_TYPE handle)
{
	btAssert(handle > 0 && handle < m_maxHandles);

	getHandle(handle)->SetNextFree(m_firstFreeHandle);
	m_firstFreeHandle = handle;

	m_numHandles--;
}


template <typename BP_FP_INT_TYPE>
BP_FP_INT_TYPE btParallelAxisSweep3Internal<BP_FP_INT_TYPE>::addHandle(const btVector3& aabbMin,const btVector3& aabbMax, void* pOwner,short int collisionFilterGroup,short int collisionFilterMask,btDispatcher* dispatcher,void* multiSapProxy)
{
	// quantize the bounds
	BP_FP_INT_TYPE min[3], max[3];
	quantize(min, aabbMin, 0);
	quantize(max, aabbMax, 1);

	// allocate a handle
	BP_FP_INT_TYPE handle = allocHandle();
	

	Handle* pHandle = getHandle(handle);

	Handle tmpHandle;
	tmpHandle.m_uniqueId = m_handles[m_frontBufferIndex].size()+1;
	tmpHandle.m_clientObject = pOwner;
	tmpHandle.m_collisionFilterGroup = collisionFilterGroup;
	tmpHandle.m_collisionFilterMask = collisionFilterMask;
	for (BP_FP_INT_TYPE axis = 0; axis < 3; axis++)
	{
		Edge minEdge;
		minEdge.m_pos = min[axis];
		minEdge.m_handle = m_handles[m_frontBufferIndex].size();
		tmpHandle.m_minEdges[axis] = m_edges[m_frontBufferIndex][axis].size();

		m_edges[m_frontBufferIndex][axis].push_back(minEdge);
		
		Edge maxEdge;
		maxEdge.m_pos = max[axis];
		maxEdge.m_handle = m_handles[m_frontBufferIndex].size();
		tmpHandle.m_maxEdges[axis] = m_edges[m_frontBufferIndex][axis].size();

		m_edges[m_frontBufferIndex][axis].push_back(maxEdge);
	}
	tmpHandle.m_multiSapParentProxy = 0;
	tmpHandle.m_aabbMin = aabbMin;
	tmpHandle.m_aabbMax = aabbMax;

	m_handles[m_frontBufferIndex].push_back(tmpHandle);

	pHandle->m_uniqueId = static_cast<int>(handle);
	//pHandle->m_pOverlaps = 0;
	pHandle->m_clientObject = pOwner;
	pHandle->m_collisionFilterGroup = collisionFilterGroup;
	pHandle->m_collisionFilterMask = collisionFilterMask;
	pHandle->m_multiSapParentProxy = multiSapProxy;

	// compute current limit of edge arrays
	BP_FP_INT_TYPE limit = static_cast<BP_FP_INT_TYPE>(m_numHandles * 2);

	
	// insert new edges just inside the max boundary edge
	for (BP_FP_INT_TYPE axis = 0; axis < 3; axis++)
	{

		m_pHandles[0].m_maxEdges[axis] += 2;

		m_pEdges[axis][limit + 1] = m_pEdges[axis][limit - 1];

		m_pEdges[axis][limit - 1].m_pos = min[axis];
		m_pEdges[axis][limit - 1].m_handle = handle;

		m_pEdges[axis][limit].m_pos = max[axis];
		m_pEdges[axis][limit].m_handle = handle;

		pHandle->m_minEdges[axis] = static_cast<BP_FP_INT_TYPE>(limit - 1);
		pHandle->m_maxEdges[axis] = limit;
	}

	// now sort the new edges to their correct position
	sortMinDown(0, pHandle->m_minEdges[0], dispatcher,false);
	sortMaxDown(0, pHandle->m_maxEdges[0], dispatcher,false);
	sortMinDown(1, pHandle->m_minEdges[1], dispatcher,false);
	sortMaxDown(1, pHandle->m_maxEdges[1], dispatcher,false);
	sortMinDown(2, pHandle->m_minEdges[2], dispatcher,true);
	sortMaxDown(2, pHandle->m_maxEdges[2], dispatcher,true);



	return handle;
}


template <typename BP_FP_INT_TYPE>
void btParallelAxisSweep3Internal<BP_FP_INT_TYPE>::removeHandle(BP_FP_INT_TYPE handle,btDispatcher* dispatcher)
{

	Handle* pHandle = getHandle(handle);

	//explicitly remove the pairs containing the proxy
	//we could do it also in the sortMinUp (passing true)
	///@todo: compare performance
	if (!m_pairCache->hasDeferredRemoval())
	{
		m_pairCache2->removeOverlappingPairsContainingProxy(pHandle,dispatcher);
	}

	// compute current limit of edge arrays
	int limit = static_cast<int>(m_numHandles * 2);
	
	int axis;

	for (axis = 0;axis<3;axis++)
	{
		m_pHandles[0].m_maxEdges[axis] -= 2;
	}

	// remove the edges by sorting them up to the end of the list
	for ( axis = 0; axis < 3; axis++)
	{
		Edge* pEdges = m_pEdges[axis];
		BP_FP_INT_TYPE max = pHandle->m_maxEdges[axis];
		pEdges[max].m_pos = m_handleSentinel;

		sortMaxUp(axis,max,dispatcher,false);


		BP_FP_INT_TYPE i = pHandle->m_minEdges[axis];
		pEdges[i].m_pos = m_handleSentinel;


		sortMinUp(axis,i,dispatcher,false);

		pEdges[limit-1].m_handle = 0;
		pEdges[limit-1].m_pos = m_handleSentinel;
		
#ifdef DEBUG_BROADPHASE
			debugPrintAxis(axis,false);
#endif //DEBUG_BROADPHASE


	}


	// free the handle
	freeHandle(handle);

	
}

template <typename BP_FP_INT_TYPE>
void btParallelAxisSweep3Internal<BP_FP_INT_TYPE>::resetPool(btDispatcher* /*dispatcher*/)
{
	if (m_numHandles == 0)
	{
		m_firstFreeHandle = 1;
		{
			for (BP_FP_INT_TYPE i = m_firstFreeHandle; i < m_maxHandles; i++)
				m_pHandles[i].SetNextFree(static_cast<BP_FP_INT_TYPE>(i + 1));
			m_pHandles[m_maxHandles - 1].SetNextFree(0);
		}
	}
}       


extern int gOverlappingPairs;
//#include <stdio.h>



#include "BulletCollision/Gimpact/gim_radixsort.h"
/*
template <typename BP_FP_INT_TYPE>
bool btParallelAxisSweep3Internal<BP_FP_INT_TYPE>::newTestAabbOverlap(BP_FP_INT_TYPE handleIndex, int buffer0, int buffer1, int axis0,int axis1)
{
	return false;
}
*/

template <typename BP_FP_INT_TYPE>
void	btParallelAxisSweep3Internal<BP_FP_INT_TYPE>::calculateOverlappingPairs(btDispatcher* dispatcher)
{

	int target0 = -1;
	int target1 = -1;
	static int mycounter = 0;
	mycounter++;
/*	if (mycounter==243)
	{
		printf("...\n");
	}
	printf("%d -----------------------------------------------------------\n",mycounter);
	*/
again:

	
	//did we add new handles since last buffer swap?
	int oldSize = m_handles[1-m_frontBufferIndex].size(); 
	int numHandlesAdded = m_handles[m_frontBufferIndex].size() - oldSize;

	if (numHandlesAdded)
	{
		BT_PROFILE("parallel SAP numHandlesAdded ");
		for (int i=0;i<numHandlesAdded;i++)
		{
			Handle h = m_handles[m_frontBufferIndex][oldSize+i];
			m_handles[1-m_frontBufferIndex].push_back(h);
			for (int ax = 0;ax<3;ax++)
			{
				Edge edgeMin = m_edges[m_frontBufferIndex][ax][oldSize*2+i*2];
				m_edges[1-m_frontBufferIndex][ax].push_back(edgeMin);
				Edge edgeMax = m_edges[m_frontBufferIndex][ax][oldSize*2+i*2+1];
				m_edges[1-m_frontBufferIndex][ax].push_back(edgeMax);
			}
		}
	}

	{
	

		memcopy_elements_func copyFunc;
		class EDGE_GET_KEY
		{
		public:
			inline int operator()( const Edge& a)
			{
				return a.m_pos;
			}
		};
		EDGE_GET_KEY getKey;
		//sort the edges for each of the 3 axis
		for (int ax=0;ax<3;ax++)
		{
			BT_PROFILE("parallel SAP gim_radix_sort ");

			Edge* unsortedEdges = &m_edges[1-m_frontBufferIndex][ax][0];
			Edge* sortedEdges = &m_edges[m_frontBufferIndex][ax][0];
			int numEdges = m_edges[m_frontBufferIndex][ax].size();
    		gim_radix_sort(sortedEdges,numEdges,getKey,copyFunc);
		}

		//update the handles
		for (int ax=0;ax<3;ax++)
		{
			BT_PROFILE("parallel SAP update handles ");

			for (int i=0;i<m_edges[m_frontBufferIndex][ax].size();i++)
			{
				int h = m_edges[m_frontBufferIndex][ax][i].m_handle;
				bool isMax = m_edges[m_frontBufferIndex][ax][i].IsMax();
				if (isMax)
				{
					m_handles[m_frontBufferIndex][h].m_maxEdges[ax] = i;
				} else
				{
					m_handles[m_frontBufferIndex][h].m_minEdges[ax] = i;
				}
			}
		}

		//now detect the added pairs (if any)

		btAssert(m_handles[m_frontBufferIndex].size()==m_handles[1-m_frontBufferIndex].size());
/*
		if (testAabbOverlap(&m_handles[m_frontBufferIndex][37],&m_handles[m_frontBufferIndex][41]))
		{
			btBroadphasePair* pair = m_pairCache2->findPair(&m_handles[m_frontBufferIndex][37],&m_handles[m_frontBufferIndex][41]);
			printf("uid 38 and uid 42 overlaps\n");
		}
		*/

		if (gVerbose)
		{
			printf("Debug btParallelAxisSweep3.h");

			for (int ax=0;ax<3;ax++)
			{
				printf("\nPrevious axis: %d\n", ax);
		
				for (int i=0;i<m_handles[m_frontBufferIndex].size();i++)
				{
					btAssert(m_handles[m_frontBufferIndex][i].m_clientObject == m_handles[1-m_frontBufferIndex][i].m_clientObject);
					int prevMinIndex = m_handles[1-m_frontBufferIndex][i].m_minEdges[ax];
					int prevMaxIndex = m_handles[1-m_frontBufferIndex][i].m_maxEdges[ax];
					printf("[%d:%d,%d]",i,prevMinIndex,prevMaxIndex);
				}
			}
	
			for (int ax=0;ax<3;ax++)
			{
				printf("\nCurrent Axis: %d\n", ax);
		
				for (int i=0;i<m_handles[m_frontBufferIndex].size();i++)
				{
					btAssert(m_handles[m_frontBufferIndex][i].m_clientObject == m_handles[1-m_frontBufferIndex][i].m_clientObject);
					int curMinIndex = m_handles[m_frontBufferIndex][i].m_minEdges[ax];
					int curMaxIndex = m_handles[m_frontBufferIndex][i].m_maxEdges[ax];
					printf("[%d:%d,%d]",i,curMinIndex,curMaxIndex);
				}
			}
		}
	


		for (int ax=0;ax<3;ax++)
		{
			BT_PROFILE("parallel SAP find pairs");

			for (int i=0;i<m_handles[m_frontBufferIndex].size();i++)
			{
				btAssert(m_handles[m_frontBufferIndex][i].m_clientObject == m_handles[1-m_frontBufferIndex][i].m_clientObject);
				int prevMinIndex = m_handles[1-m_frontBufferIndex][i].m_minEdges[ax];
				int curMinIndex = m_handles[m_frontBufferIndex][i].m_minEdges[ax];
				int dmin = curMinIndex-prevMinIndex;
				int prevMaxIndex = m_handles[1-m_frontBufferIndex][i].m_maxEdges[ax];
				int curMaxIndex = m_handles[m_frontBufferIndex][i].m_maxEdges[ax];
				int dmax = curMaxIndex-prevMaxIndex;

			
				if (dmin<0)
				{
					for (int j=prevMinIndex;j>curMinIndex;j--)
					{
						{
							int otherHandle = m_edges[1-m_frontBufferIndex][ax][j].m_handle;
						
							if ((i==target0 && otherHandle==target1)||(i==target1 && otherHandle==target0))
							{
								printf("checkme\n");
							}
							//skip self-collisions
							if (otherHandle != i)
							{
								const int axis1 = (1  << ax) & 3;
								const int axis2 = (1  << axis1) & 3;
		//						if (newTestOverlap2D(j,1-m_frontBufferIndex,1-m_frontBufferIndex ,axis1,axis2))
								if (slowTestAabbOverlap(&m_handles[m_frontBufferIndex][i],&m_handles[m_frontBufferIndex][otherHandle])
								//	&& !testAabbOverlap(&m_handles[1-m_frontBufferIndex][i],&m_handles[1-m_frontBufferIndex][otherHandle])
								)

								{
								
									int uidA =m_handles[m_frontBufferIndex][i].m_uniqueId;
									int uidB =m_handles[m_frontBufferIndex][otherHandle].m_uniqueId;

									if ((m_handles[m_frontBufferIndex][i].m_uniqueId==38 && m_handles[m_frontBufferIndex][otherHandle].m_uniqueId==42)||	(m_handles[m_frontBufferIndex][i].m_uniqueId==42 && m_handles[m_frontBufferIndex][otherHandle].m_uniqueId==38))
										printf("add pair (dmin<0) %d, %d\n",uidA,uidB);
								
									m_pairCache->addOverlappingPair(&m_handles[m_frontBufferIndex][i],&m_handles[m_frontBufferIndex][otherHandle]);
									if (m_userPairCallback)
										m_userPairCallback->addOverlappingPair(&m_handles[m_frontBufferIndex][i],&m_handles[m_frontBufferIndex][otherHandle]);

								}
							}
						}
					}
				}

				//add overlapping pairs
				if (dmin<0)
				{
					for (int j=prevMinIndex;j>curMinIndex;j--)
					{
						{
							int otherHandle = m_edges[m_frontBufferIndex][ax][j].m_handle;
						
							if ((i==target0 && otherHandle==target1)||(i==target1 && otherHandle==target0))
							{
								printf("checkme\n");
							}
							//skip self-collisions
							if (otherHandle != i)
							{
								const int axis1 = (1  << ax) & 3;
								const int axis2 = (1  << axis1) & 3;
		//						if (newTestOverlap2D(j,1-m_frontBufferIndex,1-m_frontBufferIndex ,axis1,axis2))
								if (slowTestAabbOverlap(&m_handles[m_frontBufferIndex][i],&m_handles[m_frontBufferIndex][otherHandle])
								//	&& !testAabbOverlap(&m_handles[1-m_frontBufferIndex][i],&m_handles[1-m_frontBufferIndex][otherHandle])
								)

								{
								
									int uidA =m_handles[m_frontBufferIndex][i].m_uniqueId;
									int uidB =m_handles[m_frontBufferIndex][otherHandle].m_uniqueId;

									if ((m_handles[m_frontBufferIndex][i].m_uniqueId==38 && m_handles[m_frontBufferIndex][otherHandle].m_uniqueId==42)||	(m_handles[m_frontBufferIndex][i].m_uniqueId==42 && m_handles[m_frontBufferIndex][otherHandle].m_uniqueId==38))
										printf("add pair (dmin<0) %d, %d\n",uidA,uidB);
								
									m_pairCache->addOverlappingPair(&m_handles[m_frontBufferIndex][i],&m_handles[m_frontBufferIndex][otherHandle]);
									if (m_userPairCallback)
										m_userPairCallback->addOverlappingPair(&m_handles[m_frontBufferIndex][i],&m_handles[m_frontBufferIndex][otherHandle]);

								}
							}
						}
					}
				}
				//add overlapping pairs
				if (dmax>0)
				{
					for (int j=prevMaxIndex;j<curMaxIndex;j++)
					{
						{
							int otherHandle = m_edges[m_frontBufferIndex][ax][j].m_handle;
							//skip self-collisions
							if (otherHandle != i)
							{
								if ((i==target0 && otherHandle==target1)||(i==target1 && otherHandle==target0))
								{
									printf("checkme\n");
								}

								const int axis1 = (1  << ax) & 3;
								const int axis2 = (1  << axis1) & 3;
		//						if (newTestOverlap2D(j,1-m_frontBufferIndex,1-m_frontBufferIndex ,axis1,axis2))
								if (slowTestAabbOverlap(&m_handles[m_frontBufferIndex][i],&m_handles[m_frontBufferIndex][otherHandle])
							//		&& testAabbOverlap(&m_handles[1-m_frontBufferIndex][i],&m_handles[1-m_frontBufferIndex][otherHandle])
									)
								{
								
	//								if (gVerbose)
									/*
									if ((m_handles[m_frontBufferIndex][i].m_uniqueId==38 && m_handles[m_frontBufferIndex][otherHandle].m_uniqueId==42)||	(m_handles[m_frontBufferIndex][i].m_uniqueId==42 && m_handles[m_frontBufferIndex][otherHandle].m_uniqueId==38))
									{
										printf("add pair (dmax>0) %d, %d\n",m_handles[m_frontBufferIndex][i].m_uniqueId, m_handles[m_frontBufferIndex][otherHandle].m_uniqueId);
									}
									*/

									m_pairCache->addOverlappingPair(&m_handles[m_frontBufferIndex][i],&m_handles[m_frontBufferIndex][otherHandle]);
									if (m_userPairCallback)
										m_userPairCallback->addOverlappingPair(&m_handles[m_frontBufferIndex][i],&m_handles[m_frontBufferIndex][otherHandle]);

								}
							}
						}
					}
				}

				//add overlapping pairs
				if (dmax>0)
				{
					for (int j=prevMaxIndex;j<curMaxIndex;j++)
					{
						{
							int otherHandle = m_edges[1-m_frontBufferIndex][ax][j].m_handle;
							//skip self-collisions
							if (otherHandle != i)
							{
								if ((i==target0 && otherHandle==target1)||(i==target1 && otherHandle==target0))
								{
									printf("checkme\n");
								}

								const int axis1 = (1  << ax) & 3;
								const int axis2 = (1  << axis1) & 3;
		//						if (newTestOverlap2D(j,1-m_frontBufferIndex,1-m_frontBufferIndex ,axis1,axis2))
								if (slowTestAabbOverlap(&m_handles[m_frontBufferIndex][i],&m_handles[m_frontBufferIndex][otherHandle])
							//		&& testAabbOverlap(&m_handles[1-m_frontBufferIndex][i],&m_handles[1-m_frontBufferIndex][otherHandle])
									)
								{
								
	//								if (gVerbose)
									/*
									if ((m_handles[m_frontBufferIndex][i].m_uniqueId==38 && m_handles[m_frontBufferIndex][otherHandle].m_uniqueId==42)||	(m_handles[m_frontBufferIndex][i].m_uniqueId==42 && m_handles[m_frontBufferIndex][otherHandle].m_uniqueId==38))
									{
										printf("add pair (dmax>0) %d, %d\n",m_handles[m_frontBufferIndex][i].m_uniqueId, m_handles[m_frontBufferIndex][otherHandle].m_uniqueId);
									}
									*/
									m_pairCache->addOverlappingPair(&m_handles[m_frontBufferIndex][i],&m_handles[m_frontBufferIndex][otherHandle]);
									if (m_userPairCallback)
										m_userPairCallback->addOverlappingPair(&m_handles[m_frontBufferIndex][i],&m_handles[m_frontBufferIndex][otherHandle]);

								}
							}
						}
					}
				}
				//remove pairs
				if (dmin>0)
				{
					for (int j=prevMinIndex;j<curMinIndex;j++)
					{
						{
							int otherHandle = m_edges[m_frontBufferIndex][ax][j].m_handle;
							//skip self-collisions
							if (otherHandle != i)
							{
								const int axis1 = (1  << ax) & 3;
								const int axis2 = (1  << axis1) & 3;
		//						if (newTestOverlap2D(j,1-m_frontBufferIndex,1-m_frontBufferIndex ,axis1,axis2))
								if (!slowTestAabbOverlap(&m_handles[m_frontBufferIndex][i],&m_handles[m_frontBufferIndex][otherHandle])
								//	&& !testAabbOverlap(&m_handles[1-m_frontBufferIndex][i],&m_handles[1-m_frontBufferIndex][otherHandle])
								)

								{
	//								if (gVerbose)
									/*
									if ((m_handles[m_frontBufferIndex][i].m_uniqueId==38 && m_handles[m_frontBufferIndex][otherHandle].m_uniqueId==42)||	(m_handles[m_frontBufferIndex][i].m_uniqueId==42 && m_handles[m_frontBufferIndex][otherHandle].m_uniqueId==38))
									{
										printf("remove pair (dmin>0)  %d, %d\n",m_handles[m_frontBufferIndex][i].m_uniqueId, m_handles[m_frontBufferIndex][otherHandle].m_uniqueId);
									}
									*/
									m_pairCache->removeOverlappingPair(&m_handles[m_frontBufferIndex][i],&m_handles[m_frontBufferIndex][otherHandle],dispatcher);	
									if (m_userPairCallback)
										m_userPairCallback->removeOverlappingPair(&m_handles[m_frontBufferIndex][i],&m_handles[m_frontBufferIndex][otherHandle],dispatcher);


								}
							}
						}
					}
				}
				if (dmin>0)
				{
					for (int j=prevMinIndex;j<curMinIndex;j++)
					{
						{
							int otherHandle = m_edges[1-m_frontBufferIndex][ax][j].m_handle;
							//skip self-collisions
							if (otherHandle != i)
							{
								const int axis1 = (1  << ax) & 3;
								const int axis2 = (1  << axis1) & 3;
		//						if (newTestOverlap2D(j,1-m_frontBufferIndex,1-m_frontBufferIndex ,axis1,axis2))
								if (!slowTestAabbOverlap(&m_handles[m_frontBufferIndex][i],&m_handles[m_frontBufferIndex][otherHandle])
								//	&& !testAabbOverlap(&m_handles[1-m_frontBufferIndex][i],&m_handles[1-m_frontBufferIndex][otherHandle])
								)

								{
	//								if (gVerbose)
									/*
									if ((m_handles[m_frontBufferIndex][i].m_uniqueId==38 && m_handles[m_frontBufferIndex][otherHandle].m_uniqueId==42)||	(m_handles[m_frontBufferIndex][i].m_uniqueId==42 && m_handles[m_frontBufferIndex][otherHandle].m_uniqueId==38))
									{
										printf("remove pair (dmin>0)  %d, %d\n",m_handles[m_frontBufferIndex][i].m_uniqueId, m_handles[m_frontBufferIndex][otherHandle].m_uniqueId);
									}
									*/
									m_pairCache->removeOverlappingPair(&m_handles[m_frontBufferIndex][i],&m_handles[m_frontBufferIndex][otherHandle],dispatcher);	
									if (m_userPairCallback)
										m_userPairCallback->removeOverlappingPair(&m_handles[m_frontBufferIndex][i],&m_handles[m_frontBufferIndex][otherHandle],dispatcher);


								}
							}
						}
					}
				}
				//remove pairs
				if (dmax<0)
				{
					for (int j=prevMaxIndex;j<curMaxIndex;j--)
					{
						{
							int otherHandle = m_edges[m_frontBufferIndex][ax][j].m_handle;
							//skip self-collisions
							if (otherHandle != i)
							{
								const int axis1 = (1  << ax) & 3;
								const int axis2 = (1  << axis1) & 3;
		//						if (newTestOverlap2D(j,1-m_frontBufferIndex,1-m_frontBufferIndex ,axis1,axis2))
								if (!slowTestAabbOverlap(&m_handles[m_frontBufferIndex][i],&m_handles[m_frontBufferIndex][otherHandle])
							//		&& testAabbOverlap(&m_handles[1-m_frontBufferIndex][i],&m_handles[1-m_frontBufferIndex][otherHandle])
									)
								{
									//if (gVerbose)
									/*if ((m_handles[m_frontBufferIndex][i].m_uniqueId==38 && m_handles[m_frontBufferIndex][otherHandle].m_uniqueId==42)||	(m_handles[m_frontBufferIndex][i].m_uniqueId==42 && m_handles[m_frontBufferIndex][otherHandle].m_uniqueId==38))
									{
										printf("remove  pair (dmax<0) %d, %d\n",m_handles[m_frontBufferIndex][i].m_uniqueId, m_handles[m_frontBufferIndex][otherHandle].m_uniqueId);
									}
									*/
									m_pairCache->removeOverlappingPair(&m_handles[m_frontBufferIndex][i],&m_handles[m_frontBufferIndex][otherHandle],dispatcher);	
									if (m_userPairCallback)
										m_userPairCallback->removeOverlappingPair(&m_handles[m_frontBufferIndex][i],&m_handles[m_frontBufferIndex][otherHandle],dispatcher);

								}
							}
						}
					}
				}
				if (dmax<0)
				{
					for (int j=prevMaxIndex;j<curMaxIndex;j--)
					{
						{
							int otherHandle = m_edges[1-m_frontBufferIndex][ax][j].m_handle;
							//skip self-collisions
							if (otherHandle != i)
							{
								const int axis1 = (1  << ax) & 3;
								const int axis2 = (1  << axis1) & 3;
		//						if (newTestOverlap2D(j,1-m_frontBufferIndex,1-m_frontBufferIndex ,axis1,axis2))
								if (!slowTestAabbOverlap(&m_handles[m_frontBufferIndex][i],&m_handles[m_frontBufferIndex][otherHandle])
							//		&& testAabbOverlap(&m_handles[1-m_frontBufferIndex][i],&m_handles[1-m_frontBufferIndex][otherHandle])
									)
								{
									//if (gVerbose)
									/*if ((m_handles[m_frontBufferIndex][i].m_uniqueId==38 && m_handles[m_frontBufferIndex][otherHandle].m_uniqueId==42)||	(m_handles[m_frontBufferIndex][i].m_uniqueId==42 && m_handles[m_frontBufferIndex][otherHandle].m_uniqueId==38))
									{
										printf("remove  pair (dmax<0) %d, %d\n",m_handles[m_frontBufferIndex][i].m_uniqueId, m_handles[m_frontBufferIndex][otherHandle].m_uniqueId);
									}
									*/
									m_pairCache->removeOverlappingPair(&m_handles[m_frontBufferIndex][i],&m_handles[m_frontBufferIndex][otherHandle],dispatcher);	
									if (m_userPairCallback)
										m_userPairCallback->removeOverlappingPair(&m_handles[m_frontBufferIndex][i],&m_handles[m_frontBufferIndex][otherHandle],dispatcher);

								}
							}
						}
					}
				}
			}
		}

	}
	


	if (m_pairCache->hasDeferredRemoval())
	{
	
		btBroadphasePairArray&	overlappingPairArray = m_pairCache->getOverlappingPairArray();

		//perform a sort, to find duplicates and to sort 'invalid' pairs to the end
		overlappingPairArray.quickSort(btBroadphasePairSortPredicate());

		overlappingPairArray.resize(overlappingPairArray.size() - m_invalidPair);
		m_invalidPair = 0;

		
		int i;

		btBroadphasePair previousPair;
		previousPair.m_pProxy0 = 0;
		previousPair.m_pProxy1 = 0;
		previousPair.m_algorithm = 0;
		
		
		for (i=0;i<overlappingPairArray.size();i++)
		{
		
			btBroadphasePair& pair = overlappingPairArray[i];

			bool isDuplicate = (pair == previousPair);

			previousPair = pair;

			bool needsRemoval = false;

			if (!isDuplicate)
			{
				///important to use an AABB test that is consistent with the broadphase
				bool hasOverlap = testAabbOverlap(pair.m_pProxy0,pair.m_pProxy1);

				if (hasOverlap)
				{
					needsRemoval = false;//callback->processOverlap(pair);
				} else
				{
					needsRemoval = true;
				}
			} else
			{
				//remove duplicate
				needsRemoval = true;
				//should have no algorithm
				btAssert(!pair.m_algorithm);
			}
			
			if (needsRemoval)
			{
				m_pairCache->cleanOverlappingPair(pair,dispatcher);

		//		m_overlappingPairArray.swap(i,m_overlappingPairArray.size()-1);
		//		m_overlappingPairArray.pop_back();
				pair.m_pProxy0 = 0;
				pair.m_pProxy1 = 0;
				m_invalidPair++;
				gOverlappingPairs--;
			} 
			
		}

	///if you don't like to skip the invalid pairs in the array, execute following code:
	#define CLEAN_INVALID_PAIRS 1
	#ifdef CLEAN_INVALID_PAIRS

		//perform a sort, to sort 'invalid' pairs to the end
		overlappingPairArray.quickSort(btBroadphasePairSortPredicate());

		overlappingPairArray.resize(overlappingPairArray.size() - m_invalidPair);
		m_invalidPair = 0;
	#endif//CLEAN_INVALID_PAIRS
		
		//printf("overlappingPairArray.size()=%d\n",overlappingPairArray.size());
	}

	static int count = 0;
	if (m_pairCache->getNumOverlappingPairs() != m_pairCache2->getNumOverlappingPairs())
	{
	//	printf("difference number %d: expected %d got %d !\n",count,m_pairCache2->getNumOverlappingPairs(),m_pairCache->getNumOverlappingPairs());
		count++;
	}

	//btBroadphaseProxy* p0 = &m_handles[m_frontBufferIndex][38];
	//btBroadphaseProxy* p1 = &m_handles[m_frontBufferIndex][42];

	if (0)
	{
		btBroadphasePair* pair = m_pairCache2->findPair(&m_handles[m_frontBufferIndex][38],&m_handles[m_frontBufferIndex][42]);
	btBroadphasePair* pair2 =m_pairCache->findPair(pair->m_pProxy0,pair->m_pProxy1); 
		if (!(pair2))
		{
			printf("wtf\n");
		}
	}

	{
		BT_PROFILE("parallel SAP validity check");

		for (int i=0;i<m_pairCache2->getNumOverlappingPairs();i++)
		{
			btBroadphasePair* pair = &m_pairCache2->getOverlappingPairArrayPtr()[i];
			btBroadphaseProxy* pp0 = pair->m_pProxy0;
			btBroadphaseProxy* pp1 = pair->m_pProxy1;

			if (!(m_pairCache->findPair(pp0,pp1)))
			{
				printf("pair (%d,%d) missing\n",pair->m_pProxy0->getUid(),pair->m_pProxy1->getUid());
				target0 = pair->m_pProxy0->getUid()-1;
				target1 = pair->m_pProxy1->getUid()-1;

	//			goto again;
			}
		}
	}

	{
		BT_PROFILE("parallel SAP swap buffer");

		//now sync and swap m_frontBufferIndex
		for (int i=0;i<m_handles[m_frontBufferIndex].size();i++)
		{
			m_handles[1-m_frontBufferIndex][i] = m_handles[m_frontBufferIndex][i];
		}

		for (int ax=0;ax<3;ax++)
		{
			for (int i=0;i<m_edges[m_frontBufferIndex][ax].size();i++)
			{
				m_edges[1-m_frontBufferIndex][ax][i] = m_edges[m_frontBufferIndex][ax][i];
			}
		}
	}
	m_frontBufferIndex = 1-m_frontBufferIndex;
}
#include "LinearMath/btAabbUtil2.h"

template <typename BP_FP_INT_TYPE>
bool btParallelAxisSweep3Internal<BP_FP_INT_TYPE>::slowTestAabbOverlap(btBroadphaseProxy* proxy0,btBroadphaseProxy* proxy1)
{
	return testAabbOverlap(proxy0,proxy1);
	const Handle* pHandleA = static_cast<Handle*>(proxy0);
	const Handle* pHandleB = static_cast<Handle*>(proxy1);
	bool overlap = TestAabbAgainstAabb2(pHandleA->m_aabbMin,pHandleA->m_aabbMax,pHandleB->m_aabbMin,pHandleB->m_aabbMax);
	return overlap;

	//optimization 1: check the array index (memory address), instead of the m_pos

/*	for (int axis = 0; axis < 3; axis++)
	{ 
		if (pHandleA->m_maxEdges[axis].m_pos < pHandleB->m_minEdges[axis].m_pos || 
			pHandleB->m_maxEdges[axis].m_pos < pHandleA->m_minEdges[axis].m_pos) 
		{ 
			return false; 
		} 
	}
	*/

	//return true;
}


template <typename BP_FP_INT_TYPE>
bool btParallelAxisSweep3Internal<BP_FP_INT_TYPE>::testAabbOverlap(btBroadphaseProxy* proxy0,btBroadphaseProxy* proxy1)
{
	const Handle* pHandleA = static_cast<Handle*>(proxy0);
	const Handle* pHandleB = static_cast<Handle*>(proxy1);
	
	//optimization 1: check the array index (memory address), instead of the m_pos

	for (int axis = 0; axis < 3; axis++)
	{ 
		if (pHandleA->m_maxEdges[axis] < pHandleB->m_minEdges[axis] || 
			pHandleB->m_maxEdges[axis] < pHandleA->m_minEdges[axis]) 
		{ 
			return false; 
		} 
	} 
	return true;
}

template <typename BP_FP_INT_TYPE>
bool btParallelAxisSweep3Internal<BP_FP_INT_TYPE>::testOverlap2D(const Handle* pHandleA, const Handle* pHandleB,int axis0,int axis1)
{
	//optimization 1: check the array index (memory address), instead of the m_pos

	if (pHandleA->m_maxEdges[axis0] < pHandleB->m_minEdges[axis0] || 
		pHandleB->m_maxEdges[axis0] < pHandleA->m_minEdges[axis0] ||
		pHandleA->m_maxEdges[axis1] < pHandleB->m_minEdges[axis1] ||
		pHandleB->m_maxEdges[axis1] < pHandleA->m_minEdges[axis1]) 
	{ 
		return false; 
	} 
	return true;
}

template <typename BP_FP_INT_TYPE>
void btParallelAxisSweep3Internal<BP_FP_INT_TYPE>::updateHandle(BP_FP_INT_TYPE handle, const btVector3& aabbMin,const btVector3& aabbMax,btDispatcher* dispatcher)
{
//	btAssert(bounds.IsFinite());
	//btAssert(bounds.HasVolume());

	Handle* pHandle = getHandle(handle);

	// quantize the new bounds
	BP_FP_INT_TYPE min[3], max[3];
	quantize(min, aabbMin, 0);
	quantize(max, aabbMax, 1);

	m_handles[m_frontBufferIndex][handle-1].m_aabbMin = aabbMin;
	m_handles[m_frontBufferIndex][handle-1].m_aabbMax = aabbMax;

	// update changed edges
	for (int axis = 0; axis < 3; axis++)
	{
		BP_FP_INT_TYPE emin = m_handles[m_frontBufferIndex][handle-1].m_minEdges[axis];
		BP_FP_INT_TYPE emax = m_handles[m_frontBufferIndex][handle-1].m_maxEdges[axis];

		m_edges[m_frontBufferIndex][axis][emin].m_pos = min[axis];
		m_edges[m_frontBufferIndex][axis][emax].m_pos = max[axis];
	}
	// update changed edges
	for (int axis = 0; axis < 3; axis++)
	{
		BP_FP_INT_TYPE emin = pHandle->m_minEdges[axis];
		BP_FP_INT_TYPE emax = pHandle->m_maxEdges[axis];

		int dmin = (int)min[axis] - (int)m_pEdges[axis][emin].m_pos;
		int dmax = (int)max[axis] - (int)m_pEdges[axis][emax].m_pos;

		m_pEdges[axis][emin].m_pos = min[axis];
		m_pEdges[axis][emax].m_pos = max[axis];

		
		// expand (only adds overlaps)
		if (dmin < 0)
			sortMinDown(axis, emin,dispatcher,true);

		if (dmax > 0)
			sortMaxUp(axis, emax,dispatcher,true);

		// shrink (only removes overlaps)
		if (dmin > 0)
			sortMinUp(axis, emin,dispatcher,true);

		if (dmax < 0)
			sortMaxDown(axis, emax,dispatcher,true);

#ifdef DEBUG_BROADPHASE
	debugPrintAxis(axis);
#endif //DEBUG_BROADPHASE
	}

	
}




// sorting a min edge downwards can only ever *add* overlaps
template <typename BP_FP_INT_TYPE>
void btParallelAxisSweep3Internal<BP_FP_INT_TYPE>::sortMinDown(int axis, BP_FP_INT_TYPE edge, btDispatcher* /* dispatcher */, bool updateOverlaps)
{

	Edge* pEdge = m_pEdges[axis] + edge;
	Edge* pPrev = pEdge - 1;
	Handle* pHandleEdge = getHandle(pEdge->m_handle);

	while (pEdge->m_pos < pPrev->m_pos)
	{
		Handle* pHandlePrev = getHandle(pPrev->m_handle);

		if (pPrev->IsMax())
		{
			// if previous edge is a maximum check the bounds and add an overlap if necessary
			const int axis1 = (1  << axis) & 3;
			const int axis2 = (1  << axis1) & 3;
			if (updateOverlaps && testOverlap2D(pHandleEdge, pHandlePrev,axis1,axis2))
			{
				m_pairCache2->addOverlappingPair(pHandleEdge,pHandlePrev);
				if (m_userPairCallback)
					m_userPairCallback->addOverlappingPair(pHandleEdge,pHandlePrev);
				//AddOverlap(pEdge->m_handle, pPrev->m_handle);

			}

			// update edge reference in other handle
			pHandlePrev->m_maxEdges[axis]++;
		}
		else
			pHandlePrev->m_minEdges[axis]++;

		pHandleEdge->m_minEdges[axis]--;

		// swap the edges
		Edge swap = *pEdge;
		*pEdge = *pPrev;
		*pPrev = swap;

		// decrement
		pEdge--;
		pPrev--;
	}

#ifdef DEBUG_BROADPHASE
	debugPrintAxis(axis);
#endif //DEBUG_BROADPHASE

}

// sorting a min edge upwards can only ever *remove* overlaps
template <typename BP_FP_INT_TYPE>
void btParallelAxisSweep3Internal<BP_FP_INT_TYPE>::sortMinUp(int axis, BP_FP_INT_TYPE edge, btDispatcher* dispatcher, bool updateOverlaps)
{
	Edge* pEdge = m_pEdges[axis] + edge;
	Edge* pNext = pEdge + 1;
	Handle* pHandleEdge = getHandle(pEdge->m_handle);

	while (pNext->m_handle && (pEdge->m_pos >= pNext->m_pos))
	{
		Handle* pHandleNext = getHandle(pNext->m_handle);

		if (pNext->IsMax())
		{
			Handle* handle0 = getHandle(pEdge->m_handle);
			Handle* handle1 = getHandle(pNext->m_handle);
			const int axis1 = (1  << axis) & 3;
			const int axis2 = (1  << axis1) & 3;
			
			// if next edge is maximum remove any overlap between the two handles
			if (updateOverlaps 
#ifdef USE_OVERLAP_TEST_ON_REMOVES
				&& testOverlap2D(handle0,handle1,axis1,axis2)
#endif //USE_OVERLAP_TEST_ON_REMOVES
				)
			{
				
				m_pairCache2->removeOverlappingPair(handle0,handle1,dispatcher);	
				if (m_userPairCallback)
					m_userPairCallback->removeOverlappingPair(handle0,handle1,dispatcher);
			}


			// update edge reference in other handle
			pHandleNext->m_maxEdges[axis]--;
		}
		else
			pHandleNext->m_minEdges[axis]--;

		pHandleEdge->m_minEdges[axis]++;

		// swap the edges
		Edge swap = *pEdge;
		*pEdge = *pNext;
		*pNext = swap;

		// increment
		pEdge++;
		pNext++;
	}


}

// sorting a max edge downwards can only ever *remove* overlaps
template <typename BP_FP_INT_TYPE>
void btParallelAxisSweep3Internal<BP_FP_INT_TYPE>::sortMaxDown(int axis, BP_FP_INT_TYPE edge, btDispatcher* dispatcher, bool updateOverlaps)
{

	Edge* pEdge = m_pEdges[axis] + edge;
	Edge* pPrev = pEdge - 1;
	Handle* pHandleEdge = getHandle(pEdge->m_handle);

	while (pEdge->m_pos < pPrev->m_pos)
	{
		Handle* pHandlePrev = getHandle(pPrev->m_handle);

		if (!pPrev->IsMax())
		{
			// if previous edge was a minimum remove any overlap between the two handles
			Handle* handle0 = getHandle(pEdge->m_handle);
			Handle* handle1 = getHandle(pPrev->m_handle);
			const int axis1 = (1  << axis) & 3;
			const int axis2 = (1  << axis1) & 3;

			if (updateOverlaps  
#ifdef USE_OVERLAP_TEST_ON_REMOVES
				&& testOverlap2D(handle0,handle1,axis1,axis2)
#endif //USE_OVERLAP_TEST_ON_REMOVES
				)
			{
				//this is done during the overlappingpairarray iteration/narrowphase collision

				m_pairCache2->removeOverlappingPair(handle0,handle1,dispatcher);
				if (m_userPairCallback)
					m_userPairCallback->removeOverlappingPair(handle0,handle1,dispatcher);


			}

			// update edge reference in other handle
			pHandlePrev->m_minEdges[axis]++;;
		}
		else
			pHandlePrev->m_maxEdges[axis]++;

		pHandleEdge->m_maxEdges[axis]--;

		// swap the edges
		Edge swap = *pEdge;
		*pEdge = *pPrev;
		*pPrev = swap;

		// decrement
		pEdge--;
		pPrev--;
	}

	
#ifdef DEBUG_BROADPHASE
	debugPrintAxis(axis);
#endif //DEBUG_BROADPHASE

}

// sorting a max edge upwards can only ever *add* overlaps
template <typename BP_FP_INT_TYPE>
void btParallelAxisSweep3Internal<BP_FP_INT_TYPE>::sortMaxUp(int axis, BP_FP_INT_TYPE edge, btDispatcher* /* dispatcher */, bool updateOverlaps)
{
	Edge* pEdge = m_pEdges[axis] + edge;
	Edge* pNext = pEdge + 1;
	Handle* pHandleEdge = getHandle(pEdge->m_handle);

	while (pNext->m_handle && (pEdge->m_pos >= pNext->m_pos))
	{
		Handle* pHandleNext = getHandle(pNext->m_handle);

		const int axis1 = (1  << axis) & 3;
		const int axis2 = (1  << axis1) & 3;

		if (!pNext->IsMax())
		{
			// if next edge is a minimum check the bounds and add an overlap if necessary
			if (updateOverlaps && testOverlap2D(pHandleEdge, pHandleNext,axis1,axis2))
			{
				Handle* handle0 = getHandle(pEdge->m_handle);
				Handle* handle1 = getHandle(pNext->m_handle);

				m_pairCache2->addOverlappingPair(handle0,handle1);
				if (m_userPairCallback)
					m_userPairCallback->addOverlappingPair(handle0,handle1);
			}

			// update edge reference in other handle
			pHandleNext->m_minEdges[axis]--;
		}
		else
			pHandleNext->m_maxEdges[axis]--;

		pHandleEdge->m_maxEdges[axis]++;

		// swap the edges
		Edge swap = *pEdge;
		*pEdge = *pNext;
		*pNext = swap;

		// increment
		pEdge++;
		pNext++;
	}
	
}



////////////////////////////////////////////////////////////////////

///The btParallelAxisSweep3 is a special version of a 3d axis sweep and prune that is suitable for GPU
///the idea is to sort all 3 axis every frame, and efficiently compute the 'swapping' pairs by comparing the sorted arrays with previous frame
///Each object can compute its own 'swaps' independent from other objects as follows:
///determine the begin and end point in each of the 3 axis in the previous frame and in current frame, and 
///sweep the begin and end points of each object along each of the axis from the previous location to the new location.
///objects in those ranges are considered a potential 'swap' and a AABB check is performed.
///This AABB check can be a full check, or an incremental check, to avoid adding too many duplicate pairs
///On GPU, or parallel CPU implementation, all objects can search its own overlapping pairs in parallel.
///The benefit over a uniform grid is that the 3 axis SAP can deal with large object differences.
///The benefit of a 3d SAP with incremental add/remove pairs over the 1 axis SAP with full pair search is 
///that 3 axis SAP has better performance for scenes with high spatial coherence.
///A hybrid between 3d axis SAP and 1 axis SAP is very promising: 
///use 3 axis SAP for large objects and for high spatial coherence and use 1-axis SAP in bad performing cases (initialization etc)
class btParallelAxisSweep3 : public btParallelAxisSweep3Internal<unsigned short int>
{
public:

	btParallelAxisSweep3(const btVector3& worldAabbMin,const btVector3& worldAabbMax, unsigned short int maxHandles = 16384, btOverlappingPairCache* pairCache = 0, bool disableRaycastAccelerator = false);

};



#endif//BT_PARALLEL_AXIS_SWEEP_3_H

