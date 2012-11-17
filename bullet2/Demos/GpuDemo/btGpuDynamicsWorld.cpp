#include "btGpuDynamicsWorld.h"
#include "BulletDynamics/Dynamics/btRigidBody.h"

#include "../../../opencl/gpu_rigidbody_pipeline2/CLPhysicsDemo.h"
#include "../../../opencl/gpu_rigidbody_pipeline/btGpuNarrowPhaseAndSolver.h"
#include "BulletCollision/CollisionShapes/btPolyhedralConvexShape.h"
#include "BulletCollision/CollisionShapes/btBvhTriangleMeshShape.h"


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
	int gpuShapeIndex = -1;

	int index = m_uniqueShapes.findLinearSearch(body->getCollisionShape());
	if (index==m_uniqueShapes.size())
	{
		if (body->getCollisionShape()->isPolyhedral())
		{
			m_uniqueShapes.push_back(body->getCollisionShape());

			btPolyhedralConvexShape* convex = (btPolyhedralConvexShape*)body->getCollisionShape();
			int numVertices=convex->getNumVertices();
		
			int strideInBytes=sizeof(btVector3);
			btAlignedObjectArray<btVector3> tmpVertices;
			tmpVertices.resize(numVertices);
			for (int i=0;i<numVertices;i++)
				convex->getVertex(i,tmpVertices[i]);
			const float scaling[4]={1,1,1,1};
			bool noHeightField=true;
		
			gpuShapeIndex = m_gpuPhysics->registerCollisionShape(&tmpVertices[0].getX(), strideInBytes, numVertices, scaling, noHeightField);
			m_uniqueShapeMapping.push_back(gpuShapeIndex);
		} else
		{
			if (body->getCollisionShape()->getShapeType()==TRIANGLE_MESH_SHAPE_PROXYTYPE)
			{
				m_uniqueShapes.push_back(body->getCollisionShape());

				btBvhTriangleMeshShape* trimesh = (btBvhTriangleMeshShape*) body->getCollisionShape();
				btStridingMeshInterface* meshInterface = trimesh->getMeshInterface();
				btAlignedObjectArray<btVector3> vertices;
				btAlignedObjectArray<int> indices;

				btVector3 trimeshScaling(1,1,1);
				for (int partId=0;partId<meshInterface->getNumSubParts();partId++)
				{
		
					const unsigned char *vertexbase = 0;
					int numverts = 0;
					PHY_ScalarType type = PHY_INTEGER;
					int stride = 0;
					const unsigned char *indexbase = 0;
					int indexstride = 0;
					int numfaces = 0;
					PHY_ScalarType indicestype = PHY_INTEGER;
					//PHY_ScalarType indexType=0;

					btVector3 triangleVerts[3];
					meshInterface->getLockedReadOnlyVertexIndexBase(&vertexbase,numverts,	type,stride,&indexbase,indexstride,numfaces,indicestype,partId);
					btVector3 aabbMin,aabbMax;

					for (int triangleIndex = 0 ; triangleIndex < numfaces;triangleIndex++)
					{
						unsigned int* gfxbase = (unsigned int*)(indexbase+triangleIndex*indexstride);

						for (int j=2;j>=0;j--)
						{

							int graphicsindex = indicestype==PHY_SHORT?((unsigned short*)gfxbase)[j]:gfxbase[j];
							if (type == PHY_FLOAT)
							{
								float* graphicsbase = (float*)(vertexbase+graphicsindex*stride);
								triangleVerts[j] = btVector3(
									graphicsbase[0]*trimeshScaling.getX(),
									graphicsbase[1]*trimeshScaling.getY(),
									graphicsbase[2]*trimeshScaling.getZ());
							}
							else
							{
								double* graphicsbase = (double*)(vertexbase+graphicsindex*stride);
								triangleVerts[j] = btVector3( btScalar(graphicsbase[0]*trimeshScaling.getX()), 
									btScalar(graphicsbase[1]*trimeshScaling.getY()), 
									btScalar(graphicsbase[2]*trimeshScaling.getZ()));
							}
						}
						vertices.push_back(triangleVerts[0]);
						vertices.push_back(triangleVerts[1]);
						vertices.push_back(triangleVerts[2]);
						indices.push_back(indices.size());
						indices.push_back(indices.size());
						indices.push_back(indices.size());
					}
				}
				//GraphicsShape* gfxShape = 0;//btBulletDataExtractor::createGraphicsShapeFromWavefrontObj(objData);

				//GraphicsShape* gfxShape = btBulletDataExtractor::createGraphicsShapeFromConvexHull(&sUnitSpherePoints[0],MY_UNITSPHERE_POINTS);
				float meshScaling[4] = {1,1,1,1};
				//int shapeIndex = renderer.registerShape(gfxShape->m_vertices,gfxShape->m_numvertices,gfxShape->m_indices,gfxShape->m_numIndices);
				float groundPos[4] = {0,0,0,0};

				//renderer.registerGraphicsInstance(shapeIndex,groundPos,rotOrn,color,meshScaling);
				if (vertices.size() && indices.size())
				{
					gpuShapeIndex = m_gpuPhysics->registerConcaveMesh(&vertices,&indices, meshScaling);
					m_uniqueShapeMapping.push_back(gpuShapeIndex);
				} else
				{
					printf("Error: no vertices in mesh in btGpuDynamicsWorld::addRigidBody\n");
					index = -1;
					btAssert(0);
				}
			

			} else
			{
				printf("Error: unsupported shape type in btGpuDynamicsWorld::addRigidBody\n");
				index = -1;
				btAssert(0);
			}
		}

	}

	if (index>=0)
	{
		int gpuShapeIndex= m_uniqueShapeMapping[index];
		float mass = body->getInvMass() ? 1.f/body->getInvMass() : 0.f;
		btVector3 pos = body->getWorldTransform().getOrigin();
		btQuaternion orn = body->getWorldTransform().getRotation();
	
		m_gpuPhysics->registerPhysicsInstance(mass,&pos.getX(),&orn.getX(),gpuShapeIndex,m_collisionObjects.size());

		m_collisionObjects.push_back(body);
	}
}

void	btGpuDynamicsWorld::removeCollisionObject(btCollisionObject* colObj)
{
	btDynamicsWorld::removeCollisionObject(colObj);
}


