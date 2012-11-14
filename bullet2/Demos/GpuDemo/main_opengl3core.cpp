
#include "GpuDemo.h"

#ifdef __APPLE__
#include "MacOpenGLWindow.h"
#else
#include "../rendering/rendertest/Win32OpenGLWindow.h"
#include "../rendering/rendertest/GLPrimitiveRenderer.h"
#endif
#include "../rendering/rendertest/GLInstancingRenderer.h"
#include "../../DemosCommon/OpenGL3CoreRenderer.h"
#include "LinearMath/btQuickProf.h"
#include "btGpuDynamicsWorld.h"

int g_OpenGLWidth=1024;
int g_OpenGLHeight = 768;

btgWindowInterface* window=0;

void MyKeyboardCallback(int key, int state)
{
	if (key==BTG_ESCAPE && window)
	{
		window->setRequestExit();
	}
	btDefaultKeyboardCallback(key,state);
}


#include "../rendering/rendertest/OpenGLInclude.h"

#include "../opencl/gpu_rigidbody_pipeline/CommandLineArgs.h"

void Usage()
{
	printf("\nprogram.exe [--cl_device=<int>] [--benchmark] [--disable_opencl] [--cl_platform=<int>]  [--x_dim=<int>] [--y_dim=<num>] [--z_dim=<int>] [--x_gap=<float>] [--y_gap=<float>] [--z_gap=<float>]\n"); 
};


void	DumpSimulationTime(FILE* f)
{
	CProfileIterator* profileIterator = CProfileManager::Get_Iterator();

	profileIterator->First();
	if (profileIterator->Is_Done())
		return;

	float accumulated_time=0,parent_time = profileIterator->Is_Root() ? CProfileManager::Get_Time_Since_Reset() : profileIterator->Get_Current_Parent_Total_Time();
	int i;
	int frames_since_reset = CProfileManager::Get_Frame_Count_Since_Reset();
	
	//fprintf(f,"%.3f,",	parent_time );
	float totalTime = 0.f;

	
	int numChildren = 0;
	
	for (i = 0; !profileIterator->Is_Done(); i++,profileIterator->Next())
	{
		numChildren++;
		float current_total_time = profileIterator->Get_Current_Total_Time();
		accumulated_time += current_total_time;
		float fraction = parent_time > SIMD_EPSILON ? (current_total_time / parent_time) * 100 : 0.f;
		if (!strcmp(profileIterator->Get_Current_Name(),"stepSimulation"))
		{
			fprintf(f,"%.3f,\n",current_total_time);
		}
		totalTime += current_total_time;
		//recurse into children
	}

	
	
	
	CProfileManager::Release_Iterator(profileIterator);


}
extern const char* g_deviceName;

int main(int argc, char* argv[])
{
	CommandLineArgs args(argc,argv);
	GpuDemo::ConstructionInfo ci;

	if (args.CheckCmdLineFlag("help"))
	{
		Usage();
		return 0;
	}

	bool benchmark=args.CheckCmdLineFlag("benchmark");
	bool dump_timings=args.CheckCmdLineFlag("dump_timings");
	ci.useOpenCL =!args.CheckCmdLineFlag("disable_opencl");
	ci.preferredOpenCLPlatformIndex=1;
	args.GetCmdLineArgument("cl_device", ci.preferredOpenCLDeviceIndex);
	args.GetCmdLineArgument("cl_platform", ci.preferredOpenCLPlatformIndex);
	args.GetCmdLineArgument("x_dim", ci.arraySizeX);
	args.GetCmdLineArgument("y_dim", ci.arraySizeY);
	args.GetCmdLineArgument("z_dim", ci.arraySizeZ);
	args.GetCmdLineArgument("x_gap", ci.gapX);
	args.GetCmdLineArgument("y_gap", ci.gapY);
	args.GetCmdLineArgument("z_gap", ci.gapZ);
	printf("Demo settings:\n");
	printf("x_dim=%d, y_dim=%d, z_dim=%d\n",ci.arraySizeX,ci.arraySizeY,ci.arraySizeZ);
	printf("x_gap=%f, y_gap=%f, z_gap=%f\n",ci.gapX,ci.gapY,ci.gapZ);
	
	printf("Preferred cl_device index %d\n", ci.preferredOpenCLDeviceIndex);
	printf("Preferred cl_platform index%d\n", ci.preferredOpenCLPlatformIndex);
	printf("-----------------------------------------------------\n");
	
	#ifndef BT_NO_PROFILE
	CProfileManager::Reset();
#endif //BT_NO_PROFILE


	bool syncOnly = false;

#ifdef __APPLE__
	window = new MacOpenGLWindow();
#else
	window = new Win32OpenGLWindow();
#endif
	btgWindowConstructionInfo wci(g_OpenGLWidth,g_OpenGLHeight);
	
	window->createWindow(wci);
	window->setWindowTitle("MyTest");
	printf("-----------------------------------------------------\n");

	glClearColor(1,0,0,1);
	glClear(GL_COLOR_BUFFER_BIT);
	window->startRendering();
	glFinish();
	
//	GLPrimitiveRenderer prim(g_OpenGLWidth,g_OpenGLHeight);
	float color[4] = {1,1,1,1};
//	prim.drawRect(0,0,200,200,color);
	window->endRendering();
	glFinish();

	static bool once=true;
#ifdef _WIN32
	glewInit();
#endif
	
	once=false;

	OpenGL3CoreRenderer render;
	
	glClearColor(0,1,0,1);
	glClear(GL_COLOR_BUFFER_BIT);
	
	window->endRendering();

	glFinish();

	
	window->setKeyboardCallback(MyKeyboardCallback);
	window->setMouseButtonCallback(btDefaultMouseButtonCallback);
	window->setMouseMoveCallback(btDefaultMouseMoveCallback);
	window->setWheelCallback(btDefaultWheelCallback);


	

	{
		GpuDemo* demo = new GpuDemo;
		demo->myinit();
		bool useGpu = false;
		
		

		demo->initPhysics(ci);
		render.init();
		printf("-----------------------------------------------------\n");
		
		FILE* f = 0;
		if (benchmark)
		{
			char fileName[1024];
			sprintf(fileName,"%s_%d_%d_%d.txt",g_deviceName,ci.arraySizeX,ci.arraySizeY,ci.arraySizeZ);
			printf("Open file %s\n", fileName);


			f=fopen(fileName,"w");
			if (f)
				fprintf(f,"%s (%dx%dx%d=%d),\n",  g_deviceName,ci.arraySizeX,ci.arraySizeY,ci.arraySizeZ,ci.arraySizeX*ci.arraySizeY*ci.arraySizeZ);
		}

		printf("-----------------------------------------------------\n");
		do
		{

			
			window->startRendering();
			glClearColor(0.6,0.6,0.6,1);
			glClear(GL_COLOR_BUFFER_BIT| GL_DEPTH_BUFFER_BIT|GL_STENCIL_BUFFER_BIT);
			glEnable(GL_DEPTH_TEST);
			demo->clientMoveAndDisplay();

			render.reshape(g_OpenGLWidth,g_OpenGLHeight);
			if (demo->getDynamicsWorld()->getNumCollisionObjects())
			{
				btAlignedObjectArray<btCollisionObject*> arr = demo->getDynamicsWorld()->getCollisionObjectArray();
				btCollisionObject** colObjArray = &arr[0];

				render.renderPhysicsWorld(demo->getDynamicsWorld()->getNumCollisionObjects(),colObjArray, syncOnly);
				syncOnly = true;

			}
			window->endRendering();
			glFinish();


		if (dump_timings)
			CProfileManager::dumpAll();

		if (f)
		{
			static int count=0;
			
			if (count>2 && count<102)
			{
				DumpSimulationTime(f);
			}
			if (count>=102)
				window->setRequestExit();
			count++;
		}

		} while (!window->requestedExit());

		demo->exitPhysics();
		delete demo;
		if (f)
			fclose(f);
	}

	
	
	window->closeWindow();
	delete window;
	window = 0;


	return 0;
}