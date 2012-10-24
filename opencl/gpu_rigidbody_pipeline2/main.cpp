/*
Copyright (c) 2012 Advanced Micro Devices, Inc.  

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/
//Originally written by Erwin Coumans

//
//#include "vld.h"
#ifdef _WIN32
#include <GL/glew.h>
#endif

#include "GLInstancingRenderer.h"

#ifdef _WIN32
#include "Win32OpenGLWindow.h"
#else
#include "../rendertest/MacOpenGLWindow.h"
#endif

#include "CLPhysicsDemo.h"
#include "../broadphase_benchmark/btGridBroadphaseCl.h"
#include "../../opencl/gpu_rigidbody_pipeline/btGpuNarrowPhaseAndSolver.h"

#include "LinearMath/btQuickprof.h"
#include "LinearMath/btQuaternion.h"
#include "BulletDataExtractor.h"
#include "../../opencl/gpu_rigidbody_pipeline/CommandlineArgs.h"
#include "OpenGLInclude.h"

bool printStats  = true;
bool pauseSimulation = false;
bool shootObject = false;

extern int numPairsOut;
extern int numPairsTotal;
extern bool useConvexHeightfield;


#ifdef _WIN32
bool useInterop = false;//true;
#else
bool useInterop = false;
#endif

extern bool useSapGpuBroadphase;
extern int NUM_OBJECTS_X;
extern int NUM_OBJECTS_Y;
extern int NUM_OBJECTS_Z;
extern bool keepStaticObjects;
extern float X_GAP;
extern float Y_GAP;
extern float Z_GAP;

char* fileName="../../bin/1000 stack.bullet";
void Usage()
{
	printf("\nprogram.exe [--pause_simulation=<0 or 1>] [--load_bulletfile=test.bullet] [--enable_interop=<0 or 1>] [--enable_gpusap=<0 or 1>] [--enable_convexheightfield=<0 or 1>] [--enable_static=<0 or 1>] [--x_dim=<int>] [--y_dim=<num>] [--z_dim=<int>] [--x_gap=<float>] [--y_gap=<float>] [--z_gap=<float>]\n"); 
};

int main(int argc, char* argv[])
{
		
	CommandLineArgs args(argc,argv);

	if (args.CheckCmdLineFlag("help"))
	{
		Usage();
		return 0;
	}

	args.GetCmdLineArgument("enable_interop", useInterop);
	printf("useInterop=%d\n",useInterop);

	args.GetCmdLineArgument("enable_convexheightfield", useConvexHeightfield);
	printf("enable_convexheightfield=%d\n",useConvexHeightfield);

	args.GetCmdLineArgument("enable_gpusap", useSapGpuBroadphase);
	printf("enable_gpusap=%d\n",useSapGpuBroadphase);

	args.GetCmdLineArgument("pause_simulation", pauseSimulation);
	printf("pause_simulation=%d\n",pauseSimulation);
	args.GetCmdLineArgument("x_dim", NUM_OBJECTS_X);
	args.GetCmdLineArgument("y_dim", NUM_OBJECTS_Y);
	args.GetCmdLineArgument("z_dim", NUM_OBJECTS_Z);

	args.GetCmdLineArgument("x_gap", X_GAP);
	args.GetCmdLineArgument("y_gap", Y_GAP);
	args.GetCmdLineArgument("z_gap", Z_GAP);
	printf("x_dim=%d, y_dim=%d, z_dim=%d\n",NUM_OBJECTS_X,NUM_OBJECTS_Y,NUM_OBJECTS_Z);
	printf("x_gap=%f, y_gap=%f, z_gap=%f\n",X_GAP,Y_GAP,Z_GAP);
	
	args.GetCmdLineArgument("enable_static", keepStaticObjects);
	printf("enable_static=%d\n",keepStaticObjects);	

	
	char* tmpfile = 0;
	args.GetCmdLineArgument("load_bulletfile", tmpfile );
	if (tmpfile)
		fileName = tmpfile;

	printf("load_bulletfile=%s\n",fileName);

	
	printf("\n");
#ifdef _WIN32
	Win32OpenGLWindow* window = new Win32OpenGLWindow();
	btgWindowConstructionInfo wci;
	window->createWindow(wci);

#else
    MacOpenGLWindow* window = new MacOpenGLWindow();
	window->init(512,512);
#endif
    

#ifdef _WIN32
	GLenum err = glewInit();
#endif
    
	window->startRendering();
	window->endRendering();

	GLInstancingRenderer render(MAX_CONVEX_BODIES_CL);

	
		


	CLPhysicsDemo demo(window, render.getMaxShapeCapacity());
	
	
	demo.init(-1,-1,useInterop);

	render.InitShaders();

	if (useInterop)
		demo.setupInterop();

	createScene(render, demo, useConvexHeightfield,fileName);
		

	printf("numPhysicsInstances= %d\n", demo.m_numPhysicsInstances);
	printf("numDynamicPhysicsInstances= %d\n", demo.m_numDynamicPhysicsInstances);
	


	render.writeTransforms();


    
    window->startRendering();
    render.RenderScene();
    window->endRendering();

	window->setMouseMoveCallback(btDefaultMouseMoveCallback);
	window->setMouseButtonCallback(btDefaultMouseButtonCallback);
	window->setWheelCallback(btDefaultWheelCallback);
	window->setKeyboardCallback(btDefaultKeyboardCallback);
    
	while (!window->requestedExit())
	{
		CProfileManager::Reset();
		
		if (shootObject)
		{
			shootObject = false;
			
			btVector3 linVel;// = (m_cameraPosition-m_cameraTargetPosition).normalize()*-100;

			int x,y;
			window->getMouseCoordinates(x,y);
			render.getMouseDirection(&linVel[0],x,y);
			linVel.normalize();
			linVel*=100;

//			btVector3 startPos;
			demo.setObjectLinearVelocity(&linVel[0],0);
			float orn[4] = {0,0,0,1};
			float pos[4];
			render.getCameraPosition(pos);
			
//			demo.setObjectTransform(pos,orn,0);
			render.writeSingleInstanceTransformToGPU(pos,orn,0);
//			createScene(render, demo);
//			printf("numPhysicsInstances= %d\n", demo.m_numPhysicsInstances);
//			printf("numDynamicPhysicsInstances= %d\n", demo.m_numDynamicPhysicsInstances);
//			render.writeTransforms();
		}
		if (!pauseSimulation )
			demo.stepSimulation();


		window->startRendering();
		glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
		render.RenderScene();
		window->endRendering();

		{
			BT_PROFILE("glFinish");
			glFinish();
		}

		CProfileManager::Increment_Frame_Counter();

		

		
		
		 if (printStats && !pauseSimulation)
		 {
			static int count = 3;
			count--;
			if (count<0)
			{

				CProfileManager::dumpAll();
				printf("total broadphase pairs= %d\n", numPairsTotal);
				printf("numPairsOut (culled)  = %d\n", numPairsOut);
				printStats  = false;
			} else
			{
//				printf(".");
			}
		 }
		

	}

	
	demo.cleanup();

	render.CleanupShaders();
	
	window->closeWindow();
	delete window;
	
	
	
	return 0;
}