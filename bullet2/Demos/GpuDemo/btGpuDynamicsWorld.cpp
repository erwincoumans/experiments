#include "btGpuDynamicsWorld.h"
#include "BulletDynamics/Dynamics/btRigidBody.h"

#include "../../../opencl/gpu_rigidbody_pipeline2/CLPhysicsDemo.h"
#include "../../../opencl/gpu_rigidbody_pipeline/btGpuNarrowPhaseAndSolver.h"
#include "BulletCollision/CollisionShapes/btPolyhedralConvexShape.h"
#include "LinearMath/btQuickprof.h"


#ifdef _WIN32
	#include <wiNdOws.h>
#endif

struct btGpuInternalData
{

};

btGpuDynamicsWorld::btGpuDynamicsWorld(int preferredOpenCLPlatformIndex,int preferredOpenCLDeviceIndex)
:btDynamicsWorld(0,0,0),
m_gravity(0,-10,0),
m_once(true)
{
	m_gpuPhysics = new CLPhysicsDemo(512*1024, MAX_CONVEX_BODIES_CL);
	bool useInterop = false;
	///platform and device are swapped, todo: fix this and make it consistent
	m_gpuPhysics->init(preferredOpenCLDeviceIndex,preferredOpenCLPlatformIndex,useInterop);
}

btGpuDynamicsWorld::~btGpuDynamicsWorld()
{
	delete m_gpuPhysics;
}

void btGpuDynamicsWorld::exitOpenCL()
{
}






int		btGpuDynamicsWorld::stepSimulation( btScalar timeStep,int maxSubSteps, btScalar fixedTimeStep)
{
#ifndef BT_NO_PROFILE
	CProfileManager::Reset();
#endif //BT_NO_PROFILE

	BT_PROFILE("stepSimulation");

	//convert all shapes now, and if any change, reset all (todo)
	
	if (m_once)
	{
		m_once = false;
		m_gpuPhysics->writeBodiesToGpu();
	}

	m_gpuPhysics->stepSimulation();

	{
		{
			BT_PROFILE("readbackBodiesToCpu");
			//now copy info back to rigid bodies....
			m_gpuPhysics->readbackBodiesToCpu();
		}

		
		{
			BT_PROFILE("scatter transforms into rigidbody");
			for (int i=0;i<this->m_collisionObjects.size();i++)
			{
				btVector3 pos;
				btQuaternion orn;
				m_gpuPhysics->getObjectTransformFromCpu(&pos[0],&orn[0],i);
				btTransform newTrans;
				newTrans.setOrigin(pos);
				newTrans.setRotation(orn);
				this->m_collisionObjects[i]->setWorldTransform(newTrans);
			}
		}
	}


#ifndef BT_NO_PROFILE
	CProfileManager::Increment_Frame_Counter();
#endif //BT_NO_PROFILE


	return 1;
}


void	btGpuDynamicsWorld::setGravity(const btVector3& gravity)
{
}

void	btGpuDynamicsWorld::addRigidBody(btRigidBody* body)
{

	body->setMotionState(0);

	int index = m_uniqueShapes.findLinearSearch(body->getCollisionShape());
	if (index==m_uniqueShapes.size())
	{
		m_uniqueShapes.push_back(body->getCollisionShape());
		btAssert(body->getCollisionShape()->isPolyhedral());
		btPolyhedralConvexShape* convex = (btPolyhedralConvexShape*)body->getCollisionShape();
		int numVertices=convex->getNumVertices();
		
		int strideInBytes=sizeof(btVector3);
		btAlignedObjectArray<btVector3> tmpVertices;
		tmpVertices.resize(numVertices);
		for (int i=0;i<numVertices;i++)
			convex->getVertex(i,tmpVertices[i]);
		const float scaling[4]={1,1,1,1};
		bool noHeightField=true;
		
		int gpuShapeIndex = m_gpuPhysics->registerCollisionShape(&tmpVertices[0].getX(), strideInBytes, numVertices, scaling, noHeightField);
		m_uniqueShapeMapping.push_back(gpuShapeIndex);

	}

	int gpuShapeIndex= m_uniqueShapeMapping[index];
	float mass = body->getInvMass() ? 1.f/body->getInvMass() : 0.f;
	btVector3 pos = body->getWorldTransform().getOrigin();
	btQuaternion orn = body->getWorldTransform().getRotation();
	
	m_gpuPhysics->registerPhysicsInstance(mass,&pos.getX(),&orn.getX(),gpuShapeIndex,m_collisionObjects.size());

	m_collisionObjects.push_back(body);
}

void	btGpuDynamicsWorld::removeCollisionObject(btCollisionObject* colObj)
{
	btDynamicsWorld::removeCollisionObject(colObj);
}


