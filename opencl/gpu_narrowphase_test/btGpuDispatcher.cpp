#include "btGpuDispatcher.h"
#include "BulletCollision/BroadphaseCollision/btOverlappingPairCache.h"
#include "../gpu_rigidbody_pipeline2/ConvexHullContact.h"

btGpuDispatcher::btGpuDispatcher(btCollisionConfiguration* collisionConfiguration, cl_context ctx,cl_device_id device, cl_command_queue  q)
	:btCollisionDispatcher(collisionConfiguration)
{
	m_satCollision = new GpuSatCollision(ctx,device,q);
}

btGpuDispatcher::~btGpuDispatcher()
{
	delete m_satCollision;
}


void	btGpuDispatcher::dispatchAllCollisionPairs(btOverlappingPairCache* pairCache,const btDispatcherInfo& dispatchInfo,btDispatcher* dispatcher)
{
	int numPairs = pairCache->getNumOverlappingPairs();
	btBroadphasePair* pairs = pairCache->getOverlappingPairArrayPtr();

}