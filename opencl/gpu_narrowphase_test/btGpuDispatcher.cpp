//#pragma optimize( "gty",off )

#include "btGpuDispatcher.h"
#include "BulletCollision/BroadphaseCollision/btOverlappingPairCache.h"
#include "../gpu_rigidbody_pipeline2/ConvexHullContact.h"
#include "BulletCollision/CollisionShapes/btConvexHullShape.h"
#include "BulletCollision/BroadphaseCollision/btCollisionAlgorithm.h"
#include "LinearMath/btQuickprof.h"


btGpuDispatcher::btGpuDispatcher(btCollisionConfiguration* collisionConfiguration, cl_context ctx,cl_device_id device, cl_command_queue  q)
	:btCollisionDispatcher(collisionConfiguration),
	m_ctx(ctx), m_device(device), m_queue(q)
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
//		btAssert(0);
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
	BT_PROFILE("btGpuDispatcher::dispatchAllCollisionPairs");

	int numBulletPairs = pairCache->getNumOverlappingPairs();
	if (!numBulletPairs)
		return;

	btBroadphasePair* bulletPairs = pairCache->getOverlappingPairArrayPtr();
	btAlignedObjectArray<int> pairMapping;
	//leave this empty, it is for unused heightfields
	btAlignedObjectArray<ChNarrowphase::ShapeData> shapeBuf;
	
	btAlignedObjectArray<int2> hostPairs;
	hostPairs.reserve(numBulletPairs);

	btAlignedObjectArray<RigidBodyBase::Body> bodyBuf;
	bodyBuf.reserve(8192);

	btAlignedObjectArray<Contact4> contactOut;
	int numContacts=0;
	ChNarrowphase::Config cfg;

	{
		BT_PROFILE("process numBulletPairs");
		for (int i=0;i<numBulletPairs;i++)
		{
			btBroadphasePair& collisionPair = bulletPairs[i];
			btCollisionObject* colObj0 = (btCollisionObject*)collisionPair.m_pProxy0->m_clientObject;
			btCollisionObject* colObj1 = (btCollisionObject*)collisionPair.m_pProxy1->m_clientObject;

			if (!colObj0->getCollisionShape()->isPolyhedral() || !colObj1->getCollisionShape()->isPolyhedral())
				continue;

			//can this 'needsCollision' be computed on GPU?
			bool needs =false;
			{
//				BT_PROFILE("needsCollision");
				needs = needsCollision(colObj0,colObj1);
			}
			if (needs)
			{
	//			BT_PROFILE("needs");
		
				//dispatcher will keep algorithms persistent in the collision pair
				if (!collisionPair.m_algorithm)
				{
		//			BT_PROFILE("PersistentManifoldCachingAlgorithm");
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

					{
						//BT_PROFILE("refreshContactPoints");
						mf->m_manifoldPtr->refreshContactPoints(colObj0->getWorldTransform(),colObj1->getWorldTransform());
					}

					//idea is to convert data to GPU friendly

					int bodyIndexA=-1;
					int bodyIndexB=-1;
	

					{
				//		BT_PROFILE("bodyBuf.push_bacj A");
						RigidBodyBase::Body body0 = createBodyFromBulletCollisionObject(colObj0);
						body0.m_collidableIdx = -1;
						///we need to set the invmass, because there is a filter in the kernel for static versus static (both objects have zero (inv)mass )
						body0.m_invMass = 1;//colObj0->isStaticOrKinematicObject() ? 0 : 1;
						bodyIndexA = bodyBuf.size();
						bodyBuf.push_back(body0);

					}
					{
					//	BT_PROFILE("bodyBuf.push_back B");

						RigidBodyBase::Body body1 = createBodyFromBulletCollisionObject(colObj1);
						body1.m_collidableIdx = -1;
						body1.m_invMass = 1;//colObj1->isStaticOrKinematicObject() ? 0 : 1;
						bodyIndexB= bodyBuf.size();
						bodyBuf.push_back(body1);
					}

				

				
				



					if (colObj0->getCollisionShape()->isPolyhedral())
					{

						btPolyhedralConvexShape* convexHull0 = (btPolyhedralConvexShape*)colObj0->getCollisionShape();
						const btConvexPolyhedron* c = convexHull0->getConvexPolyhedron();

						if (c->m_gpuCollidableIndex<0)
						{
							BT_PROFILE("serializeConvexHull A");

							c->m_gpuCollidableIndex = m_hostCollidables.size();
							btCollidable& col = m_hostCollidables.expand();
							col.m_shapeType = CollisionShape::SHAPE_CONVEX_HULL;
							col.m_shapeIndex = m_hostConvexData.size();
							ConvexPolyhedronCL& convex0 = m_hostConvexData.expand();
						//	BT_PROFILE("serializeConvexHull");
							serializeConvexHull(convexHull0, convex0,m_vertices,m_uniqueEdges,m_faces,m_indices);
						}
						bodyBuf[bodyIndexA].m_collidableIdx = c->m_gpuCollidableIndex;
					}


					if (colObj1->getCollisionShape()->isPolyhedral())
					{

						btPolyhedralConvexShape* convexHull1 = (btPolyhedralConvexShape*)colObj1->getCollisionShape();
						const btConvexPolyhedron* c = convexHull1->getConvexPolyhedron();

						if (c->m_gpuCollidableIndex<0)
						{
							BT_PROFILE("serializeConvexHull B");

							c->m_gpuCollidableIndex = m_hostCollidables.size();
							btCollidable& col = m_hostCollidables.expand();
							col.m_shapeType = CollisionShape::SHAPE_CONVEX_HULL;
							col.m_shapeIndex = m_hostConvexData.size();
							ConvexPolyhedronCL& convex1 = m_hostConvexData.expand();
							//BT_PROFILE("serializeConvexHull");
							serializeConvexHull(convexHull1, convex1,m_vertices,m_uniqueEdges,m_faces,m_indices);
						}
						bodyBuf[bodyIndexB].m_collidableIdx = c->m_gpuCollidableIndex;
					}

					{
					//	BT_PROFILE("hostPairs.push_back");
						int2 pair;
						pair.x = bodyIndexA;
						pair.y = bodyIndexB;
						hostPairs.push_back(pair);
						pairMapping.push_back(i);

	/*					BT_PROFILE("computeConvexConvexContactsGPUSATSingle");

						m_satCollision->computeConvexConvexContactsGPUSATSingle(
								bodyIndexA,bodyIndexB,
								bodyBuf[bodyIndexA].m_collidableIdx,bodyBuf[bodyIndexB].m_collidableIdx,
								&bodyBuf,&shapeBuf,
								&contactOut,numContacts,cfg,
								m_hostConvexData,m_hostConvexData,
								m_vertices,m_uniqueEdges,m_faces,m_indices,
								m_vertices,m_uniqueEdges,m_faces,m_indices,
								m_hostCollidables,m_hostCollidables);
								*/

					}
				
	//				printf("numContacts = %d\n", numContacts);
					}
				}
		}
	}

	bool sequential=false;
	if (sequential)
	{
	for (int i=0;i<hostPairs.size();i++)
	{
		int bodyIndexA = hostPairs[i].x;
		int bodyIndexB = hostPairs[i].y;

		int curContactOut = numContacts;

		m_satCollision->computeConvexConvexContactsGPUSATSingle(
							bodyIndexA,bodyIndexB,
							bodyBuf[bodyIndexA].m_collidableIdx,bodyBuf[bodyIndexB].m_collidableIdx,
							&bodyBuf,&shapeBuf,
							&contactOut,numContacts,cfg,
							m_hostConvexData,m_hostConvexData,
							m_vertices,m_uniqueEdges,m_faces,m_indices,
							m_vertices,m_uniqueEdges,m_faces,m_indices,
							m_hostCollidables,m_hostCollidables);
		if (curContactOut !=numContacts)
		{
			contactOut[curContactOut].m_batchIdx = i;//?
		}
	}
	}
	else
	{
	btOpenCLArray<int2> gpuPairs(m_ctx,m_queue);
	gpuPairs.copyFromHost(hostPairs);

	btOpenCLArray<RigidBodyBase::Body> gpuBodyBuf(m_ctx,m_queue);
	gpuBodyBuf.copyFromHost(bodyBuf);

	btOpenCLArray<ChNarrowphase::ShapeData> gpuShapeBuf(m_ctx,m_queue);
	btOpenCLArray<Contact4> gpuContacts(m_ctx,m_queue,1024*1024);

	btOpenCLArray<ConvexPolyhedronCL>	gpuConvexData(m_ctx,m_queue);
	gpuConvexData.copyFromHost(m_hostConvexData);

	btOpenCLArray<btVector3>				gpuVertices(m_ctx,m_queue);
	gpuVertices.copyFromHost(m_vertices);
	btOpenCLArray<btVector3>				gpuUniqueEdges(m_ctx,m_queue);
	gpuUniqueEdges.copyFromHost(m_uniqueEdges);
	btOpenCLArray<btGpuFace>				gpuFaces(m_ctx,m_queue);
	gpuFaces.copyFromHost(m_faces);
	btOpenCLArray<int>						gpuIndices(m_ctx,m_queue);
	gpuIndices.copyFromHost(m_indices);

	btOpenCLArray<btCollidable>			gpuCollidables(m_ctx,m_queue);
	gpuCollidables.copyFromHost(m_hostCollidables);

	btOpenCLArray<btYetAnotherAabb> clAabbs(m_ctx,m_queue,1);
	btOpenCLArray<int4> triangleConvexPairs(m_ctx,m_queue,1);

	int numObjects = 0;
	int maxTriConvexPairCapacity = 0;
	int numTriConvexPairsOut = 0;

	static bool useGPU= true;
	if (useGPU)
	{
		BT_PROFILE("computeConvexConvexContactsGPUSAT_AND_OpenCLArrays");

		 btOpenCLArray<float4> worldVertsB1GPU(m_ctx,m_queue);
		btOpenCLArray<int4> clippingFacesOutGPU(m_ctx,m_queue,1);
		btOpenCLArray<float4> worldNormalsAGPU(m_ctx,m_queue);
		btOpenCLArray<float4> worldVertsA1GPU(m_ctx,m_queue);
		btOpenCLArray<float4> worldVertsB2GPU(m_ctx,m_queue);


		m_satCollision->computeConvexConvexContactsGPUSAT(
			&gpuPairs,gpuPairs.size(),&gpuBodyBuf,&gpuShapeBuf,
			&gpuContacts,numContacts,cfg,gpuConvexData,gpuVertices,gpuUniqueEdges,gpuFaces,gpuIndices,
			gpuCollidables,clAabbs,
			worldVertsB1GPU,clippingFacesOutGPU,worldNormalsAGPU,worldVertsA1GPU,worldVertsB2GPU,
			numObjects,maxTriConvexPairCapacity,triangleConvexPairs,numTriConvexPairsOut);

	} else
	{
		m_satCollision->computeConvexConvexContactsGPUSAT_sequential(
			&gpuPairs,gpuPairs.size(),&gpuBodyBuf,&gpuShapeBuf,
			&gpuContacts,numContacts,cfg,gpuConvexData,gpuVertices,gpuUniqueEdges,gpuFaces,gpuIndices,
			gpuCollidables,clAabbs,numObjects,maxTriConvexPairCapacity,triangleConvexPairs,numTriConvexPairsOut);
	}

	{
		BT_PROFILE("gpuContacts.copyToHost");
		gpuContacts.copyToHost(contactOut);
	}
	}

	{
		BT_PROFILE("addContactPoint");
		//printf("numContacts = %d\n",numContacts);
		for (int i=0;i<numContacts;i++)
		{
			int newPairIndex = contactOut[i].getBatchIdx();
			int pairIndex = pairMapping[newPairIndex];
			btBroadphasePair& collisionPair = bulletPairs[pairIndex];
			btCollisionObject* colObj0 = (btCollisionObject*)collisionPair.m_pProxy0->m_clientObject;
			btCollisionObject* colObj1 = (btCollisionObject*)collisionPair.m_pProxy1->m_clientObject;

			btAssert(collisionPair.m_algorithm);
			PersistentManifoldCachingAlgorithm* mf = (PersistentManifoldCachingAlgorithm*) collisionPair.m_algorithm;


			btCollisionObjectWrapper obj0Wrap(0,colObj0->getCollisionShape(),colObj0,colObj0->getWorldTransform());
			btCollisionObjectWrapper obj1Wrap(0,colObj1->getCollisionShape(),colObj1,colObj1->getWorldTransform());
			btManifoldResult contactPointResult(&obj0Wrap,&obj1Wrap);
			contactPointResult.setPersistentManifold(mf->m_manifoldPtr);


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

	btCollisionDispatcher::dispatchAllCollisionPairs(pairCache,dispatchInfo,dispatcher);

}