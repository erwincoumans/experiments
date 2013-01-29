
#include "GpuDemo.h"

#ifdef __APPLE__
#include "MacOpenGLWindow.h"
#else
#include "../rendering/rendertest/Win32OpenGLWindow.h"
#include "../rendering/rendertest/GLPrimitiveRenderer.h"
#endif
#include "../../DemosCommon/OpenGL2Renderer.h"
#include "btGpuDynamicsWorld.h"
#include "../rendering/rendertest/OpenGLInclude.h"

int g_OpenGLWidth=1024;
int g_OpenGLHeight = 768;

//cube_vbo is unused for now (future GL/CL interop)
GLuint               cube_vbo=0;

btgDefaultOpenGLWindow* window  = 0;
OpenGL2Renderer* render=0;

void MyKeyboardCallback(int key, int state)
{
	if (key==BTG_ESCAPE && window)
	{
		window ->setRequestExit();
	}
	if (render)
		render->keyboardCallback(key);
}


int main(int argc, char* argv[])
{
	
	


#ifdef __APPLE__
	window = new MacOpenGLWindow();
#else
	window = new Win32OpenGLWindow();
#endif
	btgWindowConstructionInfo wci(g_OpenGLWidth,g_OpenGLHeight);

	wci.m_openglVersion = 2;
	
	window->createWindow(wci);
	window->setWindowTitle("MyTest");
	
#ifdef _WIN32
	glewInit();
#endif
		
	render = new OpenGL2Renderer;

	window->startRendering();
	glFinish();
	glClearColor(1,0,0,1);
	glClear(GL_COLOR_BUFFER_BIT);
//	GLPrimitiveRenderer prim(g_OpenGLWidth,g_OpenGLHeight);
	float color[4] = {1,1,1,1};
//	prim.drawRect(0,0,200,200,color);
	window->endRendering();
	glFinish();

	glClearColor(0,1,0,1);
	glClear(GL_COLOR_BUFFER_BIT);
	
	window->endRendering();
	glFinish();

	GLint err = glGetError();
    btAssert(err==GL_NO_ERROR);

	window->setKeyboardCallback(MyKeyboardCallback);


	{
		GpuDemo* demo = new GpuBoxDemo;
		
		//demo->myinit();

		GLint err = glGetError();
		btAssert(err==GL_NO_ERROR);
		
		GpuDemo::ConstructionInfo ci;
		ci.useOpenCL = true;
		demo->initPhysics(ci);
		err = glGetError();
		btAssert(err==GL_NO_ERROR);
		do
		{
			GLint err = glGetError();
			btAssert(err==GL_NO_ERROR);
			
			window->startRendering();
			glClearColor(0,1,0,1);
			glClear(GL_COLOR_BUFFER_BIT| GL_DEPTH_BUFFER_BIT|GL_STENCIL_BUFFER_BIT);
			glEnable(GL_DEPTH_TEST);
			err = glGetError();
			btAssert(err==GL_NO_ERROR);
			
			demo->clientMoveAndDisplay();

			err = glGetError();
			btAssert(err==GL_NO_ERROR);
			
			render->init();
			
			err = glGetError();
			btAssert(err==GL_NO_ERROR);
			
			if (demo->getDynamicsWorld() && demo->getDynamicsWorld()->getNumCollisionObjects())
			{
				btAlignedObjectArray<btCollisionObject*> arr = demo->getDynamicsWorld()->getCollisionObjectArray();
				btCollisionObject** colObjArray = &arr[0];
				render->renderPhysicsWorld(demo->getDynamicsWorld()->getNumCollisionObjects(), colObjArray);
			}
			err = glGetError();
			btAssert(err==GL_NO_ERROR);
			
			window->endRendering();
			err = glGetError();
			btAssert(err==GL_NO_ERROR);
			
			glFinish();
			err = glGetError();
			btAssert(err==GL_NO_ERROR);
			
		} while (!window->requestedExit());

		demo->exitPhysics();
		delete demo;
	}


	window->closeWindow();
	delete render;
	render = 0;
	delete window;
	window = 0;
	
	return 0;
}