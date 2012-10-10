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
	int numBulletPairs = pairCache->getNumOverlappingPairs();
	btBroadphasePair* bulletPairs = pairCache->getOverlappingPairArrayPtr();

	for (int i=0;i<numBulletPairs;i++)
	{
		btBroadphasePair& collisionPair = bulletPairs[i];
		btCollisionObject* colObj0 = (btCollisionObject*)collisionPair.m_pProxy0->m_clientObject;
		btCollisionObject* colObj1 = (btCollisionObject*)collisionPair.m_pProxy1->m_clientObject;

		//can this 'needsCollision' be computed on GPU?
		if (needsCollision(colObj0,colObj1))
		{
			btCollisionObjectWrapper obj0Wrap(0,colObj0->getCollisionShape(),colObj0,colObj0->getWorldTransform());
			btCollisionObjectWrapper obj1Wrap(0,colObj1->getCollisionShape(),colObj1,colObj1->getWorldTransform());


			//dispatcher will keep algorithms persistent in the collision pair
			if (!collisionPair.m_algorithm)
			{
				//create some special GPU collision algorithm with contact points?
			}

			if (collisionPair.m_algorithm)
			{
				btManifoldResult contactPointResult(&obj0Wrap,&obj1Wrap);

				//idea is to convert data to GPU friendly

				int bodyIndexA=0;
				int bodyIndexB=1;
				int collidableIndexA=0;
				int collidableIndexB=1;
	
				btAlignedObjectArray<RigidBodyBase::Body> bodyBuf;
				btAlignedObjectArray<ChNarrowphase::ShapeData> shapeBuf;
				btAlignedObjectArray<Contact4> contactOut;
				int numContacts=0;
				ChNarrowphase::Config cfg;
				btAlignedObjectArray<ConvexPolyhedronCL> hostConvexData;
	
				btAlignedObjectArray<btVector3> vertices;
				btAlignedObjectArray<btVector3> uniqueEdges;
				btAlignedObjectArray<btGpuFace> faces;
				btAlignedObjectArray<int> indices;

				btAlignedObjectArray<btCollidable> hostCollidables;
				btAlignedObjectArray<btYetAnotherAabb> clAabbs;
				int numObjects=2;
				int maxTriConvexPairCapacity=0;
				btAlignedObjectArray<int4> triangleConvexPairs;
				int numTriConvexPairsOut=0;
	
				//perform GPU narrowphase

				//copy contacts back to CPU and convert to btPersistentManifold

				m_satCollision->computeConvexConvexContactsGPUSATSingle(
						bodyIndexA,bodyIndexB,
						collidableIndexA,collidableIndexB,
						&bodyBuf,&shapeBuf,
						&contactOut,numContacts,cfg,
						hostConvexData,hostConvexData,
						vertices,uniqueEdges,faces,indices,
						vertices,uniqueEdges,faces,indices,
						hostCollidables,hostCollidables);
				}
			}
	//m_satCollision->computeConvexConvexContactsGPUSAT_sequential(pairs,numPairs,bodyBuf,shapeBuf,contactOut,nContacts,cfg,hostConvexData,vertices,uniqueEdges,faces,indices,gpuCollidables,clAabbs,numObjects,maxTriConvexPairCapacity,tripairs,numTriConvexPairsOut);
	}





}