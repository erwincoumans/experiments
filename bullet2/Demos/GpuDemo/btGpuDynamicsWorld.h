#ifndef BT_GPU_DYNAMICS_WORLD_H
#define BT_GPU_DYNAMICS_WORLD_H

class btVector3;
class btRigidBody;
class btCollisionObject;
struct btGpuInternalData;//use this struct to avoid 'leaking' all OpenCL headers into clients code base
class CLPhysicsDemo;

#include "LinearMath/btAlignedObjectArray.h"

class btGpuDynamicsWorld
{
	btAlignedObjectArray<class btCollisionObject*> m_bodies;
	btAlignedObjectArray<class btCollisionShape*> m_uniqueShapes;
	btAlignedObjectArray<int> m_uniqueShapeMapping;


	CLPhysicsDemo*		m_gpuPhysics;

	
	bool initOpenCL(int preferredDeviceIndex, int preferredPlatformIndex, bool useInterop);
	void exitOpenCL();
	
public:
	btGpuDynamicsWorld();

	virtual ~btGpuDynamicsWorld();

	virtual int	stepSimulation( btScalar timeStep);

	void	debugDrawWorld() {}

	void	setGravity(const btVector3& gravity);

	void	addRigidBody(btRigidBody* body);

	void	removeCollisionObject(btCollisionObject* colObj);

	int		getNumCollisionObjects() const;

	btAlignedObjectArray<class btCollisionObject*>& getCollisionObjectArray();

	const btAlignedObjectArray<class btCollisionObject*>& getCollisionObjectArray() const;

};


#endif //BT_GPU_DYNAMICS_WORLD_H
