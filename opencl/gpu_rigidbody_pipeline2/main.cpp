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
#include <GL/glew.h>

#include "GLInstancingRenderer.h"

#include "../opengl_interop/btOpenCLGLInteropBuffer.h"
#include "Win32OpenGLRenderManager.h"
#include "CLPhysicsDemo.h"
#include "../broadphase_benchmark/btGridBroadphaseCl.h"
#include "../../opencl/gpu_rigidbody_pipeline/btGpuNarrowPhaseAndSolver.h"

#include "LinearMath/btQuickprof.h"
#include "LinearMath/btQuaternion.h"
#include "BulletDataExtractor.h"


bool pauseSimulation = false;

extern int numPairsOut;
extern int numPairsTotal;


int main(int argc, char* argv[])
{
		
	Win32OpenGLWindow* window = new Win32OpenGLWindow();
		
	window->init(1024,768);
	GLenum err = glewInit();
	window->startRendering();
	window->endRendering();

	GLInstancingRenderer render;

	
		


	CLPhysicsDemo demo(window);
	
	bool useInterop = true;
	demo.init(-1,-1,useInterop);

		render.InitShaders();

		if (useInterop)
		demo.setupInterop();

	createScene(render, demo);
		

	printf("numPhysicsInstances= %d\n", demo.m_numPhysicsInstances);
	printf("numDynamicPhysicsInstances= %d\n", demo.m_numDynamicPhysicsInstances);
	


	render.writeTransforms();


	while (!window->requestedExit())
	{
		CProfileManager::Reset();
		
		if (!pauseSimulation )
			demo.stepSimulation();


		window->startRendering();
		render.RenderScene();
		window->endRendering();

		{
			BT_PROFILE("glFinish");
			glFinish();
		}

		CProfileManager::Increment_Frame_Counter();

		static bool printStats  = true;

		
		
		 if (printStats && !pauseSimulation)
		 {
			static int count = 0;
			count--;
			if (count<0)
			{
				count = 100;
				CProfileManager::dumpAll();
				printf("total broadphase pairs= %d\n", numPairsTotal);
				printf("numPairsOut (culled)  = %d\n", numPairsOut);
				//printStats  = false;
			} else
			{
				printf(".");
			}
		 }
		

	}

	
	demo.cleanup();

	render.CleanupShaders();
	window->exit();
	delete window;
	
	
	
	return 0;
}