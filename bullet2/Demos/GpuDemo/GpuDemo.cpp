/*
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2003-2006 Erwin Coumans  http://continuousphysics.com/Bullet/

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

//#define FORCE_CPU

#ifdef FORCE_CPU
#define btGpuDynamicsWorld btCpuDynamicsWorld
#endif //FORCE_CPU

///create 125 (5x5x5) dynamic object
#define ARRAY_SIZE_X 10
#define ARRAY_SIZE_Y 10
#define ARRAY_SIZE_Z 10

//maximum number of objects (and allow user to shoot additional boxes)
#define MAX_PROXIES (ARRAY_SIZE_X*ARRAY_SIZE_Y*ARRAY_SIZE_Z + 1024)

///scaling of the objects (0.1 = 20 centimeter boxes )
#define SCALING 1.
#define START_POS_X -5
#define START_POS_Y -5
#define START_POS_Z -3

#include "LinearMath/btVector3.h"

#include "GpuDemo.h"
//#include "GlutStuff.h"
///btBulletDynamicsCommon.h is the main Bullet include file, contains most common include files.
//#include "btBulletDynamicsCommon.h"
#ifdef FORCE_CPU
	#include "btCpuDynamicsWorld.h"
#else
	#include "btGpuDynamicsWorld.h"
#endif 


#include "BulletCollision/CollisionShapes/btBoxShape.h"
#include "BulletDynamics/Dynamics/btRigidBody.h"
#include "LinearMath/btDefaultMotionState.h"
#include "LinearMath/btQuickprof.h"


#include <stdio.h> //printf debugging


void GpuDemo::clientMoveAndDisplay()
{
//	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); 

	//simple dynamics world doesn't handle fixed-time-stepping
	float dt = getDeltaTimeInSeconds();
	
	///step the simulation
	if (m_dynamicsWorld)
	{
		m_dynamicsWorld->stepSimulation(dt);
		static int count=0;
		count++;
		if (count>100)
		{
			count=0;
			CProfileManager::dumpAll();
		}
	}
		
	renderme(); 


	swapBuffers();

}



void GpuDemo::displayCallback(void) {

	
	renderme();

	//optional but useful: debug drawing to detect problems
	if (m_dynamicsWorld)
		m_dynamicsWorld->debugDrawWorld();

	swapBuffers();
}





void	GpuDemo::initPhysics()
{
	setTexturing(true);
	setShadows(false);

	setCameraDistance(btScalar(SCALING*50.));

	///collision configuration contains default setup for memory, collision setup
	m_dynamicsWorld = new btGpuDynamicsWorld();
	
	m_dynamicsWorld->setGravity(btVector3(0,-10,0));

	///create a few basic rigid bodies
	btCollisionShape* groundShape = new btBoxShape(btVector3(btScalar(50.),btScalar(50.),btScalar(50.)));
//	btCollisionShape* groundShape = new btStaticPlaneShape(btVector3(0,1,0),50);
	
	m_collisionShapes.push_back(groundShape);

	btTransform groundTransform;
	groundTransform.setIdentity();
	groundTransform.setOrigin(btVector3(0,-50,0));

	//We can also use DemoApplication::localCreateRigidBody, but for clarity it is provided here:
	if (0)
	{
		btScalar mass(0.);

		//rigidbody is dynamic if and only if mass is non zero, otherwise static
		bool isDynamic = (mass != 0.f);

		btVector3 localInertia(0,0,0);
		if (isDynamic)
			groundShape->calculateLocalInertia(mass,localInertia);

		//using motionstate is recommended, it provides interpolation capabilities, and only synchronizes 'active' objects
		btDefaultMotionState* myMotionState = new btDefaultMotionState(groundTransform);
		btRigidBody::btRigidBodyConstructionInfo rbInfo(mass,myMotionState,groundShape,localInertia);
		btRigidBody* body = new btRigidBody(rbInfo);

		//add the body to the dynamics world
		m_dynamicsWorld->addRigidBody(body);
	}


	{
		//create a few dynamic rigidbodies
		// Re-using the same collision is better for memory usage and performance

		btCollisionShape* colShape = new btBoxShape(btVector3(SCALING*1,SCALING*1,SCALING*1));
		//btCollisionShape* colShape = new btSphereShape(btScalar(1.));
		m_collisionShapes.push_back(colShape);

		/// Create Dynamic Objects
		btTransform startTransform;
		startTransform.setIdentity();

	

		float start_x = START_POS_X - ARRAY_SIZE_X/2;
		float start_y = START_POS_Y;
		float start_z = START_POS_Z - ARRAY_SIZE_Z/2;

		for (int k=0;k<ARRAY_SIZE_Y;k++)
		{
			for (int i=0;i<ARRAY_SIZE_X;i++)
			{
				for(int j = 0;j<ARRAY_SIZE_Z;j++)
				{

						btScalar	mass = k==0? 0.f : 1.f;

					//rigidbody is dynamic if and only if mass is non zero, otherwise static
					bool isDynamic = (mass != 0.f);

					btVector3 localInertia(0,0,0);
					if (isDynamic)
						colShape->calculateLocalInertia(mass,localInertia);

					startTransform.setOrigin(SCALING*btVector3(
										btScalar(2.3*i + start_x),
										btScalar(20+2.0*k + start_y),
										btScalar(2.3*j + start_z)));

			
					//using motionstate is recommended, it provides interpolation capabilities, and only synchronizes 'active' objects
					btDefaultMotionState* myMotionState = new btDefaultMotionState(startTransform);
					btRigidBody::btRigidBodyConstructionInfo rbInfo(mass,myMotionState,colShape,localInertia);
					btRigidBody* body = new btRigidBody(rbInfo);
					

					m_dynamicsWorld->addRigidBody(body);
				}
			}
		}
	}


}
void	GpuDemo::clientResetScene()
{
	exitPhysics();
	initPhysics();
}
	

void	GpuDemo::exitPhysics()
{

	//cleanup in the reverse order of creation/initialization

	//remove the rigidbodies from the dynamics world and delete them
	int i;
	if (m_dynamicsWorld)
	{
		for (i=m_dynamicsWorld->getNumCollisionObjects()-1; i>=0 ;i--)
		{
			btCollisionObject* obj = m_dynamicsWorld->getCollisionObjectArray()[i];
			btRigidBody* body = btRigidBody::upcast(obj);
			if (body && body->getMotionState())
			{
				delete body->getMotionState();
			}
			m_dynamicsWorld->removeCollisionObject( obj );
			delete obj;
		}
	}

	//delete collision shapes
	for (int j=0;j<m_collisionShapes.size();j++)
	{
		btCollisionShape* shape = m_collisionShapes[j];
		delete shape;
	}
	m_collisionShapes.clear();

	delete m_dynamicsWorld;
	m_dynamicsWorld=0;
	

	
}



