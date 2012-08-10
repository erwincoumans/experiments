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

#include "ShapeData.h"

#ifndef __APPLE__
#include <GL/glew.h>
#endif

#include "physics_func.h"


#include "../../rendering/rendertest/GLInstancingRenderer.h"

#ifdef __APPLE__
#include "../../rendering/rendertest/MacOpenGLWindow.h"
#else
#include "../../rendering/rendertest/Win32OpenGLRenderManager.h"
#endif


#include "LinearMath/btQuickprof.h"
#include "LinearMath/btQuaternion.h"

#include "../../opencl/gpu_rigidbody_pipeline/CommandlineArgs.h"

bool printStats = false;
bool pauseSimulation = false;
bool shootObject = false;


bool useInterop = false;


const char* fileName="../../bin/1000 stack.bullet";
void Usage()
{
	printf("\nprogram.exe [--pause_simulation=<0 or 1>] [--load_bulletfile=test.bullet] [--enable_interop=<0 or 1>] [--enable_gpusap=<0 or 1>] [--enable_convexheightfield=<0 or 1>] [--enable_static=<0 or 1>] [--x_dim=<int>] [--y_dim=<num>] [--z_dim=<int>] [--x_gap=<float>] [--y_gap=<float>] [--z_gap=<float>]\n"); 
};


static float sLocalTime=0.f;
static float sFixedTimeStep=1.f/60.f;

void stepSimulation(float dt)
{
	int maxSubSteps = 10;
	int numSimulationSubSteps = 0;
	if (maxSubSteps)
	{
		//fixed timestep with interpolation
		sLocalTime += dt;
		if (sLocalTime >= sFixedTimeStep)
		{
			numSimulationSubSteps = int( sLocalTime / sFixedTimeStep);
			sLocalTime -= numSimulationSubSteps * sFixedTimeStep;
		}
		if (numSimulationSubSteps)
		{
			//clamp the number of substeps, to prevent simulation grinding spiralling down to a halt
			int clampedSimulationSteps = (numSimulationSubSteps > maxSubSteps)? maxSubSteps : numSimulationSubSteps;
			for (int i=0;i<clampedSimulationSteps;i++)
			{
				physics_simulate();
			}
		} 
	}
}


void graphics_from_physics(GLInstancingRenderer& renderer, bool syncTransformsOnly)
{

	int cubeShapeIndex  = -1;
	int strideInBytes = sizeof(float)*9;

	if (!syncTransformsOnly)
	{
		int numVertices = sizeof(cube_vertices)/strideInBytes;
		int numIndices = sizeof(cube_indices)/sizeof(int);
		cubeShapeIndex = renderer.registerShape(&cube_vertices[0],numVertices,cube_indices,numIndices);
	}


	int curGraphicsIndex=0;

	for(int i=0;i<physics_get_num_rigidbodies();i++) 
	{
		const PfxRigidState &state = physics_get_state(i);
		const PfxCollidable &coll = physics_get_collidable(i);
		const PfxRigidBody& body = physics_get_body(i);
	

		float color[4]={0,0,0,1};

		if (body.getMass()==0.f)
		{
			color[0]=1.f;
		} else
		{
			color[1]=1.f;
		}


		PfxTransform3 rbT(state.getOrientation(), state.getPosition());

		PfxShapeIterator itrShape(coll);
		for(int j=0;j<coll.getNumShapes();j++,++itrShape) {
			const PfxShape &shape = *itrShape;
			
	
			PfxTransform3 offsetT = shape.getOffsetTransform();
			PfxTransform3 worldT = rbT * offsetT;

			    PfxQuat ornWorld( worldT.getUpper3x3());

			switch(shape.getType()) {
				case kPfxShapeSphere:
					printf("render sphere\n");
				/*render_sphere(
					worldT,
					PfxVector3(1,1,1),
					PfxFloatInVec(shape.getSphere().m_radius));
					*/
				break;

				case kPfxShapeBox:
					{
				//	printf("render box\n");
					float cubeScaling[4] = {shape.getBox().m_half.getX(),shape.getBox().m_half.getY(),shape.getBox().m_half.getZ(),1};
					
					float rotOrn[4] = {ornWorld.getX(),ornWorld.getY(),ornWorld.getZ(),ornWorld.getW()};
					float position[4]={worldT.getTranslation().getX(),worldT.getTranslation().getY(),worldT.getTranslation().getZ(),0};
					if (!syncTransformsOnly)
					{
						renderer.registerGraphicsInstance(cubeShapeIndex,position,rotOrn,color,cubeScaling);
					}
					else
					{
						renderer.writeSingleInstanceTransformToCPU(position,rotOrn,curGraphicsIndex);
					}
					curGraphicsIndex++;
/*				render_box(
					worldT,
					PfxVector3(1,1,1),
					shape.getBox().m_half);
	*/
					break;
					}
				case kPfxShapeCapsule:

					printf("render_capsule\n");

					/*render_capsule(
					worldT,
					PfxVector3(1,1,1),
					PfxFloatInVec(shape.getCapsule().m_radius),
					PfxFloatInVec(shape.getCapsule().m_halfLen));
					*/
				break;

				case kPfxShapeCylinder:
					printf("render_cylinder\n");

/*				render_cylinder(
					worldT,
					PfxVector3(1,1,1),
					PfxFloatInVec(shape.getCylinder().m_radius),
					PfxFloatInVec(shape.getCylinder().m_halfLen));
					*/
				break;

				case kPfxShapeConvexMesh:
						printf("render_mesh\n");

					/*
				render_mesh(
					worldT,
					PfxVector3(1,1,1),
					convexMeshId);
					*/
				break;

				case kPfxShapeLargeTriMesh:
					printf("render_mesh\n");

					/*
				render_mesh(
					worldT,
					PfxVector3(1,1,1),
					landscapeMeshId);
					*/
				break;

				default:
				break;
			}
		}
	}


}

void create_graphics_from_physics_objects(GLInstancingRenderer& renderer)
{
	graphics_from_physics(renderer,false);
}

void sync_graphics_to_physics_objects(GLInstancingRenderer& renderer)
{
	graphics_from_physics(renderer,true);
}

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



	args.GetCmdLineArgument("pause_simulation", pauseSimulation);
	printf("pause_simulation=%d\n",pauseSimulation);
	

	
	char* tmpfile = 0;
	args.GetCmdLineArgument("load_bulletfile", tmpfile );
	if (tmpfile)
		fileName = tmpfile;

	printf("load_bulletfile=%s\n",fileName);

	
	printf("\n");
#ifdef __APPLE__
	MacOpenGLWindow* window = new MacOpenGLWindow();
#else
	Win32OpenGLWindow* window = new Win32OpenGLWindow();
#endif
	
	window->init(1024,768);
#ifndef __APPLE__
	GLenum err = glewInit();
#endif
    window->runMainLoop();
	window->startRendering();
	window->endRendering();

	int maxObjectCapacity=128*1024;

	GLInstancingRenderer render(maxObjectCapacity);
	render.setCameraDistance(30);
	
		
	render.InitShaders();


//	createSceneProgrammatically(render);
    

	render.writeTransforms();

    window->runMainLoop();

	physics_init();

	physics_create_scene(2);

	create_graphics_from_physics_objects(render);
	
	window->setMouseCallback(btDefaultMouseCallback);
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
			
			float orn[4] = {0,0,0,1};
			float pos[4];
			render.getCameraPosition(pos);
			
//			demo.setObjectTransform(pos,orn,0);
//			render.writeSingleTransformInstanceToCPU(pos,orn,0);
//			createScene(render, demo);
//			printf("numPhysicsInstances= %d\n", demo.m_numPhysicsInstances);
//			printf("numDynamicPhysicsInstances= %d\n", demo.m_numDynamicPhysicsInstances);
//			render.writeTransforms();
		}


	//	float deltaTime = 1.f/60.f;

        if (!pauseSimulation)
            physics_simulate();
//		stepSimulation(deltaTime);

		{
			BT_PROFILE("sync_graphics_to_physics_objects");
			sync_graphics_to_physics_objects(render);
		}

		{
			BT_PROFILE("render.writeTransforms");
			render.writeTransforms();
		}

		{
			BT_PROFILE("window->startRendering");
			window->startRendering();
		}
		{
			BT_PROFILE("render.RenderScene");
			render.RenderScene();
		}
		{
			BT_PROFILE("window->endRendering");
			window->endRendering();
		}

		{
			BT_PROFILE("glFinish");
			//glFinish();
        //    glFlush();
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
				//printStats  = false;
			} else
			{
//				printf(".");
			}
		 }
		

	}

	

	render.CleanupShaders();
	window->exit();
	delete window;
	
	
	
	return 0;
}