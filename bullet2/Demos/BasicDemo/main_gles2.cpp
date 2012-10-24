#include <Windows.h>

#include "../../DemosCommon/GLES2AngleWindow.h"
#include "BasicDemo.h"
#include "../../DemosCommon/GLES2Renderer.h"

#ifdef __APPLE__
#import <OpenGLES/EAGL.h>
#import <OpenGLES/ES1/gl.h>
#define	USE_IPHONE_SDK_JPEGLIB
#else
#include "GLES2/gl2.h"
#include "EGL/egl.h"

#endif//__APPLE__

int g_OpenGLWidth = 1024;
int g_OpenGLHeight = 768;

GLES2Renderer render;
void MyKeyboardCallback(int key, int state)
{
	render.keyboardCallback(key);
}

//int WINAPI WinMain(HINSTANCE hInstance,HINSTANCE hPrevInstance,LPSTR lpCmdLine,int nCmdShow)
int main(int argc, char* argv[])
{
	GLES2AngleWindow* window = new GLES2AngleWindow();

	window->createWindow(btgWindowConstructionInfo(g_OpenGLWidth,g_OpenGLHeight));
	
	window->setKeyboardCallback(MyKeyboardCallback);

	{
		BasicDemo* demo = new BasicDemo;
		demo->myinit();
		demo->initPhysics();
		render.init(g_OpenGLWidth,g_OpenGLHeight);

		do
		{
			window->startRendering();
			glClearColor(0,1,0,1);
			glClear(GL_COLOR_BUFFER_BIT| GL_DEPTH_BUFFER_BIT|GL_STENCIL_BUFFER_BIT);
			glEnable(GL_DEPTH_TEST);
			demo->clientMoveAndDisplay();

			
			render.draw(demo->getDynamicsWorld(),g_OpenGLWidth,g_OpenGLHeight);

//			render.reshape(g_OpenGLWidth,g_OpenGLHeight);
	//		render.renderPhysicsWorld(demo->getDynamicsWorld());
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
