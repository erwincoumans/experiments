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
#ifndef GPU_DEMO_H
#define GPU_DEMO_H


#include "LinearMath/btAlignedObjectArray.h"
#include "DemosCommon/DemoApplication.h"

class btBroadphaseInterface;
class btCollisionShape;
class btOverlappingPairCache;
class btCollisionDispatcher;
class btConstraintSolver;
struct btCollisionAlgorithmCreateFunc;
class btDefaultCollisionConfiguration;
class btGpuDynamicsWorld;

///GpuDemo is good starting point for learning the code base and porting.

class GpuDemo : public DemoApplication
{
	btGpuDynamicsWorld*	m_dynamicsWorld;

	//keep the collision shapes, for deletion/cleanup
	btAlignedObjectArray<btCollisionShape*>	m_collisionShapes;

	public:

	GpuDemo()
	{
		m_dynamicsWorld=0;
	
	}
	virtual ~GpuDemo()
	{
		exitPhysics();
	}
	void	initPhysics();

	void	exitPhysics();

	const btGpuDynamicsWorld* getDynamicsWorld() const
	{
		return m_dynamicsWorld;
	}
	virtual void clientMoveAndDisplay();

	virtual void displayCallback();
	virtual void	clientResetScene();
	
	static DemoApplication* Create()
	{
		GpuDemo* demo = new GpuDemo;
		demo->myinit();
		demo->initPhysics();
		return demo;
	}

	
};

#endif //GPU_DEMO_H

