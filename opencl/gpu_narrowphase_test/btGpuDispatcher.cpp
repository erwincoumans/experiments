#include "btGpuDispatcher.h"
#include "BulletCollision/BroadphaseCollision/btOverlappingPairCache.h"
#include "../gpu_rigidbody_pipeline2/ConvexHullContact.h"
#include "BulletCollision/CollisionShapes/btConvexHullShape.h"
#include "BulletCollision/BroadphaseCollision/btCollisionAlgorithm.h"
#include "LinearMath/btQuickprof.h"
#include "BulletCollision/CollisionShapes/btConvexPolyhedron.h"

btGpuDispatcher::btGpuDispatcher(btCollisionConfiguration* collisionConfiguration, cl_context ctx,cl_device_id device, cl_command_queue  q)
	:btCollisionDispatcher(collisionConfiguration)
{
	m_satCollision = new GpuSatCollision(ctx,device,q);
}

btGpuDispatcher::~btGpuDispatcher()
{
	delete m_satCollision;
}


RigidBodyBase::Body createBodyFromBulletCollisionObject(const btCollisionObject* obj)
{
		RigidBodyBase::Body body;
		btVector3 pos0 = obj->getWorldTransform().getOrigin();
		body.m_pos.x = pos0.getX();
		body.m_pos.y = pos0.getY();
		body.m_pos.z = pos0.getZ();
		body.m_pos.w = 0.f;
		btQuaternion orn0 = obj->getWorldTransform().getRotation();
		body.m_quat.x = orn0.getX();
		body.m_quat.y = orn0.getY();
		body.m_quat.z = orn0.getZ();
		body.m_quat.w = orn0.getW();
		return body;
}

void serializeConvexHull(btPolyhedralConvexShape* bulletHull, ConvexPolyhedronCL& convex,btAlignedObjectArray<btVector3>& vertices,
													btAlignedObjectArray<btVector3>& uniqueEdges,
													btAlignedObjectArray<btGpuFace>& faces,
													btAlignedObjectArray<int>& indices)
{
//	btConvexUtility convexPtr;
	btAlignedObjectArray<btVector3> orgVertices;
	for (int i=0;i<bulletHull->getNumVertices();i++)
	{
		btVector3 vtx;
		bulletHull->getVertex(i,vtx);
		orgVertices.push_back(vtx);
	}
		
//	convexPtr.initializePolyhedralFeatures(&orgVertices[0],orgVertices.size());
	const btConvexPolyhedron* convexPtr = bulletHull->getConvexPolyhedron();
	   
	convex.mC = convexPtr->mC;
	convex.mE = convexPtr->mE;
	convex.m_extents= convexPtr->m_extents;
	convex.m_localCenter = convexPtr->m_localCenter;
	convex.m_radius = convexPtr->m_radius;
	
	convex.m_numUniqueEdges = convexPtr->m_uniqueEdges.size();
	int edgeOffset = uniqueEdges.size();
	convex.m_uniqueEdgesOffset = edgeOffset;
	
	uniqueEdges.resize(edgeOffset+convex.m_numUniqueEdges);
    
	//convex data here
	int i;
	for ( i=0;i<convexPtr->m_uniqueEdges.size();i++)
	{
		uniqueEdges[edgeOffset+i] = convexPtr->m_uniqueEdges[i];
	}
    
	int faceOffset = faces.size();
	convex.m_faceOffset = faceOffset;
	convex.m_numFaces = convexPtr->m_faces.size();
	faces.resize(faceOffset+convex.m_numFaces);
	for (i=0;i<convexPtr->m_faces.size();i++)
	{
		faces[convex.m_faceOffset+i].m_plane.x = convexPtr->m_faces[i].m_plane[0];
		faces[convex.m_faceOffset+i].m_plane.y = convexPtr->m_faces[i].m_plane[1];
		faces[convex.m_faceOffset+i].m_plane.z = convexPtr->m_faces[i].m_plane[2];
		faces[convex.m_faceOffset+i].m_plane.w = convexPtr->m_faces[i].m_plane[3];
		int indexOffset = indices.size();
		int numIndices = convexPtr->m_faces[i].m_indices.size();
		faces[convex.m_faceOffset+i].m_numIndices = numIndices;
		faces[convex.m_faceOffset+i].m_indexOffset = indexOffset;
		indices.resize(indexOffset+numIndices);
		for (int p=0;p<numIndices;p++)
		{
			indices[indexOffset+p] = convexPtr->m_faces[i].m_indices[p];
		}
	}
    
	convex.m_numVertices = convexPtr->m_vertices.size();
	int vertexOffset = vertices.size();
	convex.m_vertexOffset =vertexOffset;
	vertices.resize(vertexOffset+convex.m_numVertices);
	for (int i=0;i<convexPtr->m_vertices.size();i++)
	{
		vertices[vertexOffset+i] = convexPtr->m_vertices[i];
	}


}


struct PersistentManifoldCachingAlgorithm : public btCollisionAlgorithm
{
	btPersistentManifold* m_manifoldPtr;

	PersistentManifoldCachingAlgorithm(btDispatcher*	dispatcher)
		:m_manifoldPtr(0)
	{
		m_dispatcher = dispatcher;
	}

	virtual ~PersistentManifoldCachingAlgorithm()
	{
		if (m_manifoldPtr && m_dispatcher)
		{
			m_dispatcher->releaseManifold(m_manifoldPtr);
		}
	}

	virtual void processCollision (const btCollisionObjectWrapper* body0Wrap,const btCollisionObjectWrapper* body1Wrap,const btDispatcherInfo& dispatchInfo,btManifoldResult* resultOut)
	{
		btAssert(0);
	}

	virtual btScalar calculateTimeOfImpact(btCollisionObject* body0,btCollisionObject* body1,const btDispatcherInfo& dispatchInfo,btManifoldResult* resultOut)
	{
		btAssert(0);
		return 0.f;
	}

	virtual	void	getAllContactManifolds(btManifoldArray&	manifoldArray)
	{
		if (m_manifoldPtr)
		{
			manifoldArray.push_back(m_manifoldPtr);
		}
	}


};

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
				int sz = sizeof(PersistentManifoldCachingAlgorithm);
				void* ptr = dispatcher->allocateCollisionAlgorithm(sz);
				PersistentManifoldCachingAlgorithm* m = new(ptr) PersistentManifoldCachingAlgorithm(dispatcher);
				collisionPair.m_algorithm = m;
				m->m_manifoldPtr = dispatcher->getNewManifold(colObj0,colObj1);

			}

			if (collisionPair.m_algorithm)
			{
				PersistentManifoldCachingAlgorithm* mf = (PersistentManifoldCachingAlgorithm*) collisionPair.m_algorithm;

				btManifoldResult contactPointResult(&obj0Wrap,&obj1Wrap);
				contactPointResult.setPersistentManifold(mf->m_manifoldPtr);
				mf->m_manifoldPtr->refreshContactPoints(colObj0->getWorldTransform(),colObj1->getWorldTransform());

				//idea is to convert data to GPU friendly

				int bodyIndexA=0;
				int bodyIndexB=1;
				int collidableIndexA=0;
				int collidableIndexB=1;
	
				btAlignedObjectArray<RigidBodyBase::Body> bodyBuf;

				{
					RigidBodyBase::Body body0 = createBodyFromBulletCollisionObject(colObj0);
					body0.m_collidableIdx = 0;
					bodyBuf.push_back(body0);
				}
				{
					RigidBodyBase::Body body1 = createBodyFromBulletCollisionObject(colObj1);
					body1.m_collidableIdx = 1;
					bodyBuf.push_back(body1);
				}

				btAlignedObjectArray<btCollidable> hostCollidables;
				{
					btCollidable& col = hostCollidables.expand();
					col.m_shapeType = CollisionShape::SHAPE_CONVEX_HULL;
					col.m_shapeIndex = 0;
				}

				{
					btCollidable& col = hostCollidables.expand();
					col.m_shapeType = CollisionShape::SHAPE_CONVEX_HULL;
					col.m_shapeIndex = 1;
				}

				//leave this empty, it is for unused heightfields
				btAlignedObjectArray<ChNarrowphase::ShapeData> shapeBuf;
				
				btAlignedObjectArray<Contact4> contactOut;
				int numContacts=0;
				ChNarrowphase::Config cfg;

				btAlignedObjectArray<ConvexPolyhedronCL> hostConvexData;
				

				btAlignedObjectArray<btVector3> vertices;
				btAlignedObjectArray<btVector3> uniqueEdges;
				btAlignedObjectArray<btGpuFace> faces;
				btAlignedObjectArray<int> indices;

				if (colObj0->getCollisionShape()->isPolyhedral())
				{
					btPolyhedralConvexShape* convexHull0=0;
					convexHull0 = (btPolyhedralConvexShape*)colObj0->getCollisionShape();
					ConvexPolyhedronCL& convex0 = hostConvexData.expand();
					BT_PROFILE("serializeConvexHull");
					serializeConvexHull(convexHull0, convex0,vertices,uniqueEdges,faces,indices);
				}


				if (colObj1->getCollisionShape()->isPolyhedral())
				{
					btPolyhedralConvexShape* convexHull1=0;
					convexHull1 = (btPolyhedralConvexShape*)colObj1->getCollisionShape();
					ConvexPolyhedronCL& convex1 = hostConvexData.expand();
					BT_PROFILE("serializeConvexHull");

					serializeConvexHull(convexHull1, convex1,vertices,uniqueEdges,faces,indices);
				}


				

//				btAlignedObjectArray<btYetAnotherAabb> clAabbs;
//				int numObjects=2;
	//			int maxTriConvexPairCapacity=0;
		//		btAlignedObjectArray<int4> triangleConvexPairs;
			//	int numTriConvexPairsOut=0;
	
				//perform GPU narrowphase

				//copy contacts back to CPU and convert to btPersistentManifold
				
				{
					BT_PROFILE("computeConvexConvexContactsGPUSATSingle");

					m_satCollision->computeConvexConvexContactsGPUSATSingle(
							bodyIndexA,bodyIndexB,
							collidableIndexA,collidableIndexB,
							&bodyBuf,&shapeBuf,
							&contactOut,numContacts,cfg,
							hostConvexData,hostConvexData,
							vertices,uniqueEdges,faces,indices,
							vertices,uniqueEdges,faces,indices,
							hostCollidables,hostCollidables);
					

					for (int i=0;i<numContacts;i++)
					{
						btVector3 normalOnBInWorld(-contactOut[i].m_worldNormal.x,-contactOut[i].m_worldNormal.y,-contactOut[i].m_worldNormal.z);
						for (int p=0;p<contactOut[i].getNPoints();p++)
						{
							btScalar depth = contactOut[i].getPenetration(p);
							
							float4 pt = contactOut[i].m_worldPos[p];
							btVector3 pointInWorld(pt.x,pt.y,pt.z);
							
							contactPointResult.addContactPoint(normalOnBInWorld,pointInWorld,depth);
						}

					}
				}		
//				printf("numContacts = %d\n", numContacts);
				}
			}
	//m_satCollision->computeConvexConvexContactsGPUSAT_sequential(pairs,numPairs,bodyBuf,shapeBuf,contactOut,nContacts,cfg,hostConvexData,vertices,uniqueEdges,faces,indices,gpuCollidables,clAabbs,numObjects,maxTriConvexPairCapacity,tripairs,numTriConvexPairsOut);
	}





}