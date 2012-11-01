
#include "GpuDemo.h"

#ifdef __APPLE__
#include "../rendering/rendertest/MacOpenGLWindow.h"
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



#include "../rendering/rendertest/OpenGLInclude.h"


int main(int argc, char* argv[])
{


#ifdef __APPLE__
	MacOpenGLWindow* window = new MacOpenGLWindow();
#else
	Win32OpenGLWindow* window = new Win32OpenGLWindow();
#endif
	btgWindowConstructionInfo wci(g_OpenGLWidth,g_OpenGLHeight);
	
	window->createWindow(wci);
	window->setWindowTitle("MyTest");
	
	glClearColor(1,0,0,1);
	glClear(GL_COLOR_BUFFER_BIT);
	window->startRendering();
	glFinish();
	
//	GLPrimitiveRenderer prim(g_OpenGLWidth,g_OpenGLHeight);
	float color[4] = {1,1,1,1};
//	prim.drawRect(0,0,200,200,color);
	window->endRendering();
	glFinish();

#ifdef _WIN32
	glewInit();
#endif
	
	OpenGL3CoreRenderer render;
	
	glClearColor(0,1,0,1);
	glClear(GL_COLOR_BUFFER_BIT);
	
	window->endRendering();

	glFinish();

	
	window->setKeyboardCallback(btDefaultKeyboardCallback);
	window->setMouseButtonCallback(btDefaultMouseButtonCallback);
	window->setMouseMoveCallback(btDefaultMouseMoveCallback);
	window->setWheelCallback(btDefaultWheelCallback);



	{
		GpuDemo* demo = new GpuDemo;
		demo->myinit();
		demo->initPhysics();
		render.init();
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
				render.renderPhysicsWorld(demo->getDynamicsWorld()->getNumCollisionObjects(),colObjArray);
			}
			window->endRendering();
			glFinish();
//			CProfileManager::dumpAll();
		} while (!window->requestedExit());

		demo->exitPhysics();
		delete demo;
	}


	window->closeWindow();
	delete window;

	return 0;
}