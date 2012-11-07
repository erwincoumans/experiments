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


///create 125 (5x5x5) dynamic object
#define ARRAY_SIZE_X 1
#define ARRAY_SIZE_Y 2
#define ARRAY_SIZE_Z 1

//maximum number of objects (and allow user to shoot additional boxes)
#define MAX_PROXIES (ARRAY_SIZE_X*ARRAY_SIZE_Y*ARRAY_SIZE_Z + 1024)

///scaling of the objects (0.1 = 20 centimeter boxes )
#define SCALING 1.
#define START_POS_X -5
#define START_POS_Y -5
#define START_POS_Z -3

#include "BasicDemo.h"
#include "GlutStuff.h"
///btBulletDynamicsCommon.h is the main Bullet include file, contains most common include files.
#include "btBulletDynamicsCommon.h"
#include "BulletCollision/Gimpact/btGImpactShape.h"
#include "BulletCollision/Gimpact/btGImpactCollisionAlgorithm.h"



#include "BulletCollision/CollisionShapes/btConvexPolyhedron.h"


#include <stdio.h> //printf debugging
#include "GLDebugDrawer.h"

static GLDebugDrawer gDebugDraw;

BasicDemo::BasicDemo()
{
	
	m_cameraTargetPosition.setValue(0,20,-15);
}

void BasicDemo::clientMoveAndDisplay()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); 

	//simple dynamics world doesn't handle fixed-time-stepping
	float ms = getDeltaTimeMicroseconds();
	
	///step the simulation
	if (m_dynamicsWorld)
	{
		m_dynamicsWorld->stepSimulation(1./10000.f,0);//ms / 1000000.f);
		//optional but useful: debug drawing
		m_dynamicsWorld->debugDrawWorld();
	}
		
	renderme(); 

	glFlush();

	swapBuffers();

}



void BasicDemo::displayCallback(void) {

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); 
	
	renderme();

	//optional but useful: debug drawing to detect problems
	if (m_dynamicsWorld)
		m_dynamicsWorld->debugDrawWorld();

	glFlush();
	swapBuffers();
}



btAlignedObjectArray<int> indices;
btAlignedObjectArray<btVector3> vertices;


void	BasicDemo::initPhysics()
{
	indices.clear();
	vertices.clear();

	setTexturing(true);
	setShadows(true);


	///collision configuration contains default setup for memory, collision setup
	m_collisionConfiguration = new btDefaultCollisionConfiguration();
	//m_collisionConfiguration->setConvexConvexMultipointIterations();

	///use the default collision dispatcher. For parallel processing you can use a diffent dispatcher (see Extras/BulletMultiThreaded)
	m_dispatcher = new	btCollisionDispatcher(m_collisionConfiguration);
	btGImpactCollisionAlgorithm::registerAlgorithm(m_dispatcher);

	m_broadphase = new btDbvtBroadphase();

	///the default constraint solver. For parallel processing you can use a different solver (see Extras/BulletMultiThreaded)
	btSequentialImpulseConstraintSolver* sol = new btSequentialImpulseConstraintSolver;
	m_solver = sol;

	m_dynamicsWorld = new btDiscreteDynamicsWorld(m_dispatcher,m_broadphase,m_solver,m_collisionConfiguration);
	m_dynamicsWorld->setDebugDrawer(&gDebugDraw);
	
	m_dynamicsWorld->setGravity(btVector3(0,-10,0));

	///create a few basic rigid bodies


	{
		//create a few dynamic rigidbodies
		// Re-using the same collision is better for memory usage and performance

		btBoxShape* colShape2 = new btBoxShape(btVector3(SCALING*1,SCALING*1,SCALING*1));
		colShape2->initializePolyhedralFeatures();
		const btConvexPolyhedron* poly = colShape2->getConvexPolyhedron();

		btTriangleIndexVertexArray* meshInterface = new btTriangleIndexVertexArray();
		btIndexedMesh* mesh = new btIndexedMesh();

		for (int v=0;v<poly->m_vertices.size();v++)
			vertices.push_back(poly->m_vertices[v]);

		for (int f=0;f<poly->m_faces.size();f++)
		{
			for (int t=2;t<poly->m_faces[f].m_indices.size();t++)
			{
				indices.push_back(poly->m_faces[f].m_indices[0]);
				indices.push_back(poly->m_faces[f].m_indices[t-1]);
				indices.push_back(poly->m_faces[f].m_indices[t]);
			}

		}

		mesh->m_numTriangles = indices.size()/3;
		mesh->m_numVertices = vertices.size();
		mesh->m_triangleIndexBase = (const unsigned char*)&indices[0];
		mesh->m_triangleIndexStride = 3*sizeof(int);
		mesh->m_vertexBase = (const unsigned char*)&vertices[0].getX();
		mesh->m_vertexStride = sizeof(btVector3);
		
		meshInterface ->addIndexedMesh(*mesh);
		

		btGImpactMeshShape* colShape = new btGImpactMeshShape(meshInterface);
		colShape->updateBound();

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
					startTransform.setOrigin(SCALING*btVector3(
										btScalar(2.0*i + start_x),
										btScalar(20+2.0*k + start_y),
										btScalar(2.0*j + start_z)));


					btScalar	mass = k==0? 0.f: 1.f;

					//rigidbody is dynamic if and only if mass is non zero, otherwise static
					bool isDynamic = (mass != 0.f);

					btVector3 localInertia(0,0,0);
					if (isDynamic)
						colShape->calculateLocalInertia(mass,localInertia);



			
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
void	BasicDemo::clientResetScene()
{
	exitPhysics();
	initPhysics();
}
	

void	BasicDemo::exitPhysics()
{

	//cleanup in the reverse order of creation/initialization

	//remove the rigidbodies from the dynamics world and delete them
	int i;
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

	//delete collision shapes
	for (int j=0;j<m_collisionShapes.size();j++)
	{
		btCollisionShape* shape = m_collisionShapes[j];
		delete shape;
	}
	m_collisionShapes.clear();

	delete m_dynamicsWorld;
	
	delete m_solver;
	
	delete m_broadphase;
	
	delete m_dispatcher;

	delete m_collisionConfiguration;

	
}




