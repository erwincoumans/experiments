#include "CustomCollisionDispatcher.h"
#include "BulletCollision/BroadphaseCollision/btCollisionAlgorithm.h"
#include "BulletCollision/CollisionDispatch/btCollisionObject.h"
#include "CustomConvexShape.h"
#include "CustomConvexPairCollision.h"
#include "LinearMath/btQuickprof.h"



#ifdef CL_PLATFORM_AMD

#include "Adl/Adl.h"
#include "Stubs/AdlMath.h"
#include "Stubs/AdlContact4.h"
#include "Stubs/AdlQuaternion.h"
#include "Stubs/ChNarrowPhase.h"


struct	CustomDispatchData
{
	adl::DeviceCL* m_ddcl;
	adl::Device* m_ddhost;
	ShapeDataType m_ShapeBuffer;
	adl::HostBuffer<int2>* m_pBufPairsCPU;
	adl::Buffer<int2>* m_pBufPairsGPU;
	adl::Buffer<Contact4>* m_pBufContactOutGPU;
	adl::HostBuffer<Contact4>* m_pBufContactOutCPU;
	adl::ChNarrowphase<adl::TYPE_CL>::Data* m_Data;
	adl::HostBuffer<RigidBodyBase::Body>* m_pBufRBodiesCPU;
	adl::Buffer<RigidBodyBase::Body>* m_pBufRBodiesGPU;
	int m_numAcceleratedShapes;
};
#endif //CL_PLATFORM_AMD

CustomCollisionDispatcher::CustomCollisionDispatcher(btCollisionConfiguration* collisionConfiguration
#ifdef CL_PLATFORM_AMD
		, cl_context context,cl_device_id device,cl_command_queue queue
#endif //CL_PLATFORM_AMD
):btCollisionDispatcher(collisionConfiguration),
m_internalData(0)
{
#ifdef CL_PLATFORM_AMD
	if (context && queue)
	{
		m_internalData = new CustomDispatchData();
		memset(m_internalData,0,sizeof(CustomDispatchData));

		adl::DeviceUtils::Config cfg;
		m_internalData->m_ddcl = new adl::DeviceCL();
		m_internalData->m_ddcl->m_deviceIdx = device;
		m_internalData->m_ddcl->m_context = context;
		m_internalData->m_ddcl->m_commandQueue = queue;
		m_internalData->m_ddcl->m_kernelManager = new adl::KernelManager;

		m_internalData->m_ddhost = adl::DeviceUtils::allocate( adl::TYPE_HOST, cfg );
		m_internalData->m_pBufPairsCPU = new adl::HostBuffer<int2>(m_internalData->m_ddhost, MAX_BROADPHASE_COLLISION_CL);
		m_internalData->m_pBufPairsGPU = new adl::Buffer<int2>(m_internalData->m_ddcl, MAX_BROADPHASE_COLLISION_CL);
		m_internalData->m_pBufContactOutGPU = new adl::Buffer<Contact4>(m_internalData->m_ddcl, MAX_BROADPHASE_COLLISION_CL);
		m_internalData->m_pBufContactOutCPU = new adl::HostBuffer<Contact4>(m_internalData->m_ddhost, MAX_BROADPHASE_COLLISION_CL);
		m_internalData->m_pBufRBodiesCPU = new adl::HostBuffer<RigidBodyBase::Body>(m_internalData->m_ddhost, MAX_CONVEX_BODIES_CL);
		m_internalData->m_pBufRBodiesGPU = new adl::Buffer<RigidBodyBase::Body>(m_internalData->m_ddhost, MAX_CONVEX_BODIES_CL);
		m_internalData->m_Data = adl::ChNarrowphase<adl::TYPE_CL>::allocate(m_internalData->m_ddcl);

		int numOfShapes = 1; // Just box at this time. 
		m_internalData->m_ShapeBuffer = adl::ChNarrowphase<adl::TYPE_CL>::allocateShapeBuffer(m_internalData->m_ddcl, MAX_CONVEX_SHAPES_CL);	
		m_internalData->m_numAcceleratedShapes = 0;
	}



#endif //CL_PLATFORM_AMD
}

CustomCollisionDispatcher::~CustomCollisionDispatcher(void)
{
#ifdef CL_PLATFORM_AMD
	if (m_internalData)
	{
		delete m_internalData->m_pBufPairsCPU;
		delete m_internalData->m_pBufPairsGPU;
		delete m_internalData->m_pBufContactOutGPU;
		delete m_internalData->m_pBufContactOutCPU;

		delete m_internalData->m_ddcl;
		adl::DeviceUtils::deallocate(m_internalData->m_ddhost);
	delete m_internalData;
	}
	
#endif //CL_PLATFORM_AMD

}


#ifdef CL_PLATFORM_AMD


RigidBodyBase::Body CreateRBodyCL(const btCollisionObject& body, int shapeIdx)
{
	RigidBodyBase::Body bodyCL;

	// position
	const btVector3& p = body.getWorldTransform().getOrigin();
	bodyCL.m_pos.x = p.getX();
	bodyCL.m_pos.y = p.getY();
	bodyCL.m_pos.z = p.getZ();
	bodyCL.m_pos.w = 0.0f;

	// quaternion
	btQuaternion q = body.getWorldTransform().getRotation();
	bodyCL.m_quat.x = q.getX();
	bodyCL.m_quat.y = q.getY();
	bodyCL.m_quat.z = q.getZ();
	bodyCL.m_quat.w = q.getW();

	// linear velocity
	bodyCL.m_linVel = make_float4(0.0f, 0.0f, 0.0f);

	// angular velocity
	bodyCL.m_angVel = make_float4(0.0f, 0.0f, 0.0f);

	// shape index
	bodyCL.m_shapeIdx = shapeIdx; 

#if 0
	// inverse mass
	bodyCL.m_invMass = body.getInvMass(); //needed for collisions?
#endif //

	// restituition coefficient
	bodyCL.m_restituitionCoeff = body.getRestitution();

	// friction coefficient
	bodyCL.m_frictionCoeff = body.getFriction();

	return bodyCL;
}
#endif //CL_PLATFORM_AMD

void CustomCollisionDispatcher::dispatchAllCollisionPairs(btOverlappingPairCache* pairCache,const btDispatcherInfo& dispatchInfo,btDispatcher* dispatcher) 
{
	BT_PROFILE("CustomCollisionDispatcher::dispatchAllCollisionPairs");
	btBroadphasePairArray& overlappingPairArray = pairCache->getOverlappingPairArray();
	bool bGPU = (m_internalData != 0);
#ifdef CL_PLATFORM_AMD
	if ( !bGPU )
#endif //CL_PLATFORM_AMD
	{
		btCollisionDispatcher::dispatchAllCollisionPairs(pairCache,dispatchInfo,dispatcher);
	}
#ifdef CL_PLATFORM_AMD
	else
	{
		{
			BT_PROFILE("refreshContactPoints");
			//----------------------------------------------------------------
			// GPU version of convex heightmap narrowphase collision detection
			//----------------------------------------------------------------
			for ( int i = 0; i < getNumManifolds(); i++ )
			{
				btPersistentManifold* manifold = getManifoldByIndexInternal(i);


				btCollisionObject* body0 = (btCollisionObject*)manifold->getBody0();
				btCollisionObject* body1 = (btCollisionObject*)manifold->getBody1();

				manifold->refreshContactPoints(body0->getWorldTransform(),body1->getWorldTransform());
			}
		}

		// OpenCL 
		int nColPairsFromBP = overlappingPairArray.size();
		btAssert(MAX_BROADPHASE_COLLISION_CL >= nColPairsFromBP);

		int maxBodyIndex = -1;

		{
			BT_PROFILE("CreateRBodyCL and GPU pairs");
			for ( int i=0; i<overlappingPairArray.size(); i++)
			{
				btAssert(i<MAX_BROADPHASE_COLLISION_CL);

				btBroadphasePair* pair = &overlappingPairArray[i];

				btCollisionObject* colObj0 = (btCollisionObject*)pair->m_pProxy0->m_clientObject;
				btCollisionObject* colObj1 = (btCollisionObject*)pair->m_pProxy1->m_clientObject;

				int bodyIndex0 = colObj0->getCompanionId();
				int bodyIndex1 = colObj1->getCompanionId();

				//keep a one-to-one mapping between Bullet and Adl broadphase pairs
				(*m_internalData->m_pBufPairsCPU)[i].x = bodyIndex0;
				(*m_internalData->m_pBufPairsCPU)[i].y = bodyIndex1;

				if (bodyIndex0>=0 && bodyIndex1>=0)
				{
					//create companion shapes (if necessary)

					btAssert(colObj0->getCollisionShape()->getShapeType() == CUSTOM_POLYHEDRAL_SHAPE_TYPE);
					btAssert(colObj1->getCollisionShape()->getShapeType() == CUSTOM_POLYHEDRAL_SHAPE_TYPE);

					CustomConvexShape* convexShape0 = (CustomConvexShape*)colObj0->getCollisionShape();
					CustomConvexShape* convexShape1 = (CustomConvexShape*)colObj1->getCollisionShape();

					if (convexShape0->m_acceleratedCompanionShapeIndex<0)
					{
						convexShape0->m_acceleratedCompanionShapeIndex = m_internalData->m_numAcceleratedShapes;
						adl::ChNarrowphase<adl::TYPE_CL>::setShape(m_internalData->m_ShapeBuffer, convexShape0->m_ConvexHeightField, convexShape0->m_acceleratedCompanionShapeIndex, 0.0f);
						m_internalData->m_numAcceleratedShapes++;
					}
					if (convexShape1->m_acceleratedCompanionShapeIndex<0)
					{
						convexShape1->m_acceleratedCompanionShapeIndex = m_internalData->m_numAcceleratedShapes;
						adl::ChNarrowphase<adl::TYPE_CL>::setShape(m_internalData->m_ShapeBuffer, convexShape1->m_ConvexHeightField, convexShape1->m_acceleratedCompanionShapeIndex, 0.0f);
						m_internalData->m_numAcceleratedShapes++;
					}

					if (bodyIndex0>maxBodyIndex)
						maxBodyIndex = bodyIndex0;
					if (bodyIndex1>maxBodyIndex)
						maxBodyIndex = bodyIndex1;

					(*m_internalData->m_pBufRBodiesCPU)[bodyIndex0] = CreateRBodyCL(*colObj0, convexShape0->m_acceleratedCompanionShapeIndex);
					(*m_internalData->m_pBufRBodiesCPU)[bodyIndex1] = CreateRBodyCL(*colObj1, convexShape0->m_acceleratedCompanionShapeIndex);
				} else
				{
					//TODO: dispatch using default dispatcher
					btAssert(0);
				}
			}
		}


		if (maxBodyIndex>=0)
		{
			int numOfConvexRBodies = maxBodyIndex+1;

			// Transfer rigid body data from CPU buffer to GPU buffer
			m_internalData->m_pBufRBodiesGPU->write(m_internalData->m_pBufRBodiesCPU->m_ptr, numOfConvexRBodies);
			adl::DeviceUtils::waitForCompletion(m_internalData->m_ddcl);

			m_internalData->m_pBufPairsGPU->write(m_internalData->m_pBufPairsCPU->m_ptr, MAX_BROADPHASE_COLLISION_CL);
			adl::DeviceUtils::waitForCompletion(m_internalData->m_ddcl);

			adl::ChNarrowphaseBase::Config cfgNP;
			cfgNP.m_collisionMargin = 0.01f;
			int nContactOut = 0;

			{
				BT_PROFILE("ChNarrowphase::execute");
				adl::ChNarrowphase<adl::TYPE_CL>::execute(m_internalData->m_Data, m_internalData->m_pBufPairsGPU, nColPairsFromBP, m_internalData->m_pBufRBodiesGPU, m_internalData->m_ShapeBuffer, m_internalData->m_pBufContactOutGPU, nContactOut, cfgNP);
				adl::DeviceUtils::waitForCompletion(m_internalData->m_ddcl);
			}

			{
				BT_PROFILE("read m_pBufContactOutGPU");
				m_internalData->m_pBufContactOutGPU->read(m_internalData->m_pBufContactOutCPU->m_ptr, nContactOut);//MAX_BROADPHASE_COLLISION_CL);
				adl::DeviceUtils::waitForCompletion(m_internalData->m_ddcl);
			}

			{
				BT_PROFILE("copy Contact4 to btPersistentManifold");
				// Now we got the narrowphase info from GPU and need to update rigid bodies with the info and go back to the original pipeline in Bullet physics. 
				for ( int i = 0; i < nContactOut; i++ )
				{
					Contact4 contact = (*m_internalData->m_pBufContactOutCPU)[i];

					int idxBodyA = contact.m_bodyAPtr;
					int idxBodyB = contact.m_bodyBPtr;

					btAssert(contact.m_batchIdx>=0);
					btAssert(contact.m_batchIdx<overlappingPairArray.size());

					btBroadphasePair* pair = &overlappingPairArray[contact.m_batchIdx];

					btCollisionObject* colObj0 = (btCollisionObject*)pair->m_pProxy0->m_clientObject;
					btCollisionObject* colObj1 = (btCollisionObject*)pair->m_pProxy1->m_clientObject;

					if (!pair->m_algorithm)
					{
						pair->m_algorithm = findAlgorithm(colObj0,colObj1,0);
					}

					btManifoldResult contactPointResult(colObj0, colObj1);


					CustomConvexConvexPairCollision* pairAlgo = (CustomConvexConvexPairCollision*) pair->m_algorithm;

					if (!pairAlgo->getManifoldPtr())
					{
						pairAlgo->createManifoldPtr(colObj0,colObj1,dispatchInfo);
					}
					
					contactPointResult.setPersistentManifold(pairAlgo->getManifoldPtr());
					
					contactPointResult.getPersistentManifold()->refreshContactPoints(colObj0->getWorldTransform(),colObj1->getWorldTransform());

					const btTransform& transA = colObj0->getWorldTransform();
					const btTransform& transB = colObj1->getWorldTransform();

					int numPoints = contact.getNPoints();

					for ( int k=0; k < numPoints; k++ )
					{
						btVector3 normalOnBInWorld(
							contact.m_worldNormal.x,
							contact.m_worldNormal.y,
							contact.m_worldNormal.z);
						btVector3 pointInWorldOnB(
							contact.m_worldPos[k].x,
							contact.m_worldPos[k].y,
							contact.m_worldPos[k].z);

						btScalar depth = contact.m_worldPos[k].w;

						if (depth<0)
						{
							const btVector3 deltaC = transB.getOrigin() - transA.getOrigin();

							normalOnBInWorld.normalize();

							if((deltaC.dot(normalOnBInWorld))>0.0f)
							{
								normalOnBInWorld= -normalOnBInWorld;

								contactPointResult.addContactPoint(normalOnBInWorld, pointInWorldOnB, depth);
							}
							else
							{
								contactPointResult.addContactPoint(normalOnBInWorld, pointInWorldOnB-normalOnBInWorld*depth, depth);
							}
						}
					}
				}
			}
		}
	}
#endif //CL_PLATFORM_AMD

}

