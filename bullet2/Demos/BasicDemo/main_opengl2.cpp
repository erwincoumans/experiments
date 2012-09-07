
#include "BasicDemo.h"

#ifdef __APPLE__
#include "MacOpenGLWindow.h"
#else
#include "../rendering/rendertest/Win32OpenGLWindow.h"
#include "../rendering/rendertest/GLPrimitiveRenderer.h"
#endif
#include "../../DemosCommon/OpenGL2Renderer.h"

int g_OpenGLWidth=1024;
int g_OpenGLHeight = 768;



OpenGL2Renderer render;
void MyKeyboardCallback(unsigned char key, int x, int y)
{
	render.keyboardCallback(key);
}

#include <Windows.h>

#include "GL/gl.h"

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
	glewInit();

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

	
	window->setKeyboardCallback(MyKeyboardCallback);


	{
		BasicDemo* demo = new BasicDemo;
		demo->myinit();
		demo->initPhysics();
		do
		{
			window->startRendering();
			glClearColor(0,1,0,1);
			glClear(GL_COLOR_BUFFER_BIT| GL_DEPTH_BUFFER_BIT|GL_STENCIL_BUFFER_BIT);
			glEnable(GL_DEPTH_TEST);
			demo->clientMoveAndDisplay();

			render.init();
			render.reshape(g_OpenGLWidth,g_OpenGLHeight);
			render.renderPhysicsWorld(demo->getDynamicsWorld());
			window->endRendering();
			glFinish();
		} while (!window->requestedExit());

		demo->exitPhysics();
		delete demo;
	}


	window->closeWindow();
	delete window;

	return 0;
}