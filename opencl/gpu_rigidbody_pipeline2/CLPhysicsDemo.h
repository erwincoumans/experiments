
#ifndef CL_PHYSICS_DEMO_H
#define CL_PHYSICS_DEMO_H

class Win32OpenGLWindow;

struct CLPhysicsDemo
{
	Win32OpenGLWindow* m_renderer;

	int m_numCollisionShapes;

	int m_numPhysicsInstances;

	struct InternalData* m_data;
	
	CLPhysicsDemo(Win32OpenGLWindow*	renderer);
	
	virtual ~CLPhysicsDemo();

	//btOpenCLGLInteropBuffer*	m_interopBuffer;
	
	void	init(int preferredDevice, int preferredPlatform, bool useInterop);
	
	void	setupInterop();

	int		registerCollisionShape(const float* vertices, int strideInBytes, int numVertices, const float* scaling);

	int		registerPhysicsInstance(float mass, const float* position, const float* orientation, int collisionShapeIndex, void* userPointer);

	void	writeVelocitiesToGpu();
	void	writeBodiesToGpu();

	void	cleanup();

	void	stepSimulation();
};

#endif//CL_PHYSICS_DEMO_H