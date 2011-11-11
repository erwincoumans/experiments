#ifndef CUSTOM_COLLISION_DISPATCHER_H
#define CUSTOM_COLLISION_DISPATCHER_H


#include "BulletCollision/CollisionDispatch/btCollisionDispatcher.h"
#include "BulletCollision/BroadphaseCollision/btOverlappingPairCache.h"


#define MAX_CONVEX_BODIES_CL 8192
#define MAX_PAIRS_PER_BODY_CL 32
#define MAX_CONVEX_SHAPES_CL 128
#define MAX_BROADPHASE_COLLISION_CL (MAX_CONVEX_BODIES_CL*MAX_PAIRS_PER_BODY_CL)



struct	CustomDispatchData;

#ifdef CL_PLATFORM_AMD
#ifdef __APPLE__
	#ifdef USE_MINICL
		#include <MiniCL/cl.h>
	#else
		#include <OpenCL/cl.h>
	#endif
#else //__APPLE__
	#ifdef USE_MINICL
		#include <MiniCL/cl.h>
	#else
		#include <CL/cl.h>
	#endif
#endif //__APPLE__
#endif

class CustomCollisionDispatcher : public btCollisionDispatcher
{
public:
	CustomCollisionDispatcher (btCollisionConfiguration* collisionConfiguration
#ifdef CL_PLATFORM_AMD
		, cl_context context = NULL,cl_device_id device = NULL,cl_command_queue queue = NULL
#endif //CL_PLATFORM_AMD
		);

	virtual ~CustomCollisionDispatcher(void);

protected:

	CustomDispatchData*	m_internalData;

	btBroadphasePair* GetPair(btBroadphasePairArray& pairArray, int idxBodyA, int idxBodyB);

public:
	virtual void dispatchAllCollisionPairs(btOverlappingPairCache* pairCache,const btDispatcherInfo& dispatchInfo,btDispatcher* dispatcher);
};

#endif //CUSTOM_COLLISION_DISPATCHER_H
