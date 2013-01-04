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
class btDynamicsWorld;

///GpuDemo is good starting point for learning the code base and porting.



class GpuDemo : public DemoApplication
{
	protected:

	btDynamicsWorld*	m_dynamicsWorld;

	//keep the collision shapes, for deletion/cleanup
	btAlignedObjectArray<btCollisionShape*>	m_collisionShapes;

	public:

	typedef class GpuDemo* (CreateFunc)();

	struct ConstructionInfo
	{
		bool useOpenCL;
		int preferredOpenCLPlatformIndex;
		int preferredOpenCLDeviceIndex;
		int arraySizeX;
		int arraySizeY;
		int arraySizeZ;
		bool m_useConcaveMesh;
		float gapX;
		float gapY;
		float gapZ;
		ConstructionInfo()
			:useOpenCL(false),//true),
			preferredOpenCLPlatformIndex(-1),
			preferredOpenCLDeviceIndex(-1),
			arraySizeX(3),
			arraySizeY(3 ),
			arraySizeZ(3),
			m_useConcaveMesh(false),
			gapX(4.3),
			gapY(6.0),
			gapZ(4.3)
		{
		}
	};

	GpuDemo()
	{
		m_dynamicsWorld=0;
	
	}
	virtual ~GpuDemo()
	{
		exitPhysics();
	}
	void	initPhysics(const ConstructionInfo& ci);

	virtual void setupScene(const ConstructionInfo& ci)=0;
	virtual const char* getName()=0;

	void	exitPhysics();

	const btDynamicsWorld* getDynamicsWorld() const
	{
		return m_dynamicsWorld;
	}
	virtual void clientMoveAndDisplay();

	virtual void displayCallback();
	//virtual void	clientResetScene();
	
	

	
};

class GpuDemo1 : public GpuDemo
{
public:
	virtual void setupScene(const ConstructionInfo& ci);
	virtual const char* getName()
	{
		return "GpuDemo1";
	}
	static GpuDemo* CreateFunc()
	{
		GpuDemo* demo = new GpuDemo1;
		return demo;
	}
};

class EmptyDemo : public GpuDemo
{
public:
	virtual void setupScene(const ConstructionInfo& ci);
	virtual const char* getName()
	{
		return "EmptyDemo";
	}
	static GpuDemo* CreateFunc()
	{
		GpuDemo* demo = new EmptyDemo;
		return demo;
	}
	
};


#endif //GPU_DEMO_H
