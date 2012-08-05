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
#ifndef __APPLE__
#include <GL/glew.h>
#endif

#include "../gpu_rigidbody_pipeline2/GLInstancingRenderer.h"

#include "../opengl_interop/btOpenCLGLInteropBuffer.h"
#ifdef __APPLE__
#include "MacOpenGLWindow.h"
#else
#include "../gpu_rigidbody_pipeline2/Win32OpenGLRenderManager.h"
#endif

#include "RenderScene.h"
#include "fontstash.h"


#include "LinearMath/btQuickprof.h"
#include "LinearMath/btQuaternion.h"

#include "../../opencl/gpu_rigidbody_pipeline/CommandlineArgs.h"

bool printStats = false;
bool pauseSimulation = false;
bool shootObject = false;


bool useInterop = false;

extern int NUM_OBJECTS_X;
extern int NUM_OBJECTS_Y;
extern int NUM_OBJECTS_Z;
extern bool keepStaticObjects;
extern float X_GAP;
extern float Y_GAP;
extern float Z_GAP;

const char* fileName="../../bin/1000 stack.bullet";
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
#ifdef __APPLE__
	MacOpenGLWindow* window = new MacOpenGLWindow();
#else
	Win32OpenGLWindow* window = new Win32OpenGLWindow();
#endif
	
	int width = 1024;
	int height=768;

	window->init(width,height);
#ifndef __APPLE__
	GLenum err = glewInit();
#endif
    window->runMainLoop();
	window->startRendering();
	window->endRendering();

	int maxObjectCapacity=128*1024;
	GLInstancingRenderer render(maxObjectCapacity);

	
		
	render.InitShaders();


	createSceneProgrammatically(render);
    

	render.writeTransforms();

    window->runMainLoop();

	window->setMouseCallback(btDefaultMouseCallback);
	window->setKeyboardCallback(btDefaultKeyboardCallback);



#ifdef _WIN32


		int done;
	struct sth_stash* stash = 0;
	FILE* fp = 0;
	int datasize;
	unsigned char* data;
	float sx,sy,dx,dy,lh;
	int droidRegular, droidItalic, droidBold, droidJapanese, dejavu;
	GLuint texture;

	stash = sth_create(256,256);//,1024);//512,512);
	if (!stash)
	{
		fprintf(stderr, "Could not create stash.\n");
		return -1;
	}

	// Load the first truetype font from memory (just because we can).
	fp = fopen("../../bin/DroidSerif-Regular.ttf", "rb");
	if (!fp) goto error_add_font;
	fseek(fp, 0, SEEK_END);
	datasize = (int)ftell(fp);
	fseek(fp, 0, SEEK_SET);
	data = (unsigned char*)malloc(datasize);
	if (data == NULL) goto error_add_font;
	fread(data, 1, datasize, fp);
	fclose(fp);
	fp = 0;
	
	if (!(droidRegular = sth_add_font_from_memory(stash, data)))
		goto error_add_font;

	// Load the remaining truetype fonts directly.
	if (!(droidItalic = sth_add_font(stash,"../../bin/DroidSerif-Italic.ttf")))
		goto error_add_font;
	if (!(droidBold = sth_add_font(stash,"../../bin/DroidSerif-Bold.ttf")))
		goto error_add_font;
	if (!(droidJapanese = sth_add_font(stash,"../../bin/DroidSansJapanese.ttf")))
		goto error_add_font;
#endif//_WIN32


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
			render.writeSingleInstanceTransformToGPU(pos,orn,0);
//			createScene(render, demo);
//			printf("numPhysicsInstances= %d\n", demo.m_numPhysicsInstances);
//			printf("numDynamicPhysicsInstances= %d\n", demo.m_numDynamicPhysicsInstances);
//			render.writeTransforms();
		}



		window->startRendering();
		render.RenderScene();

			glUseProgram(0);
	glBindBuffer(GL_ARRAY_BUFFER,0);
	glBindVertexArray(0);
//	glFinish();
	glClearColor(0,0,0,1);
#ifdef _WIN32
	if (1)
	{
		BT_PROFILE("font stash rendering");
				// Update and render
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glDisable(GL_TEXTURE_2D);
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		glOrtho(0,width,0,height,-1,1);
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		glDisable(GL_DEPTH_TEST);
		glColor4ub(255,0,0,255);
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);

		sx = 0; sy = height-96;
		
		sth_begin_draw(stash);
		
		dx = sx; dy = sy;
		static int once=0;

		for (int i=0;i<1;i++)
		{
			dx = sx;
			if (once!=1)
			{
				//need to save this file as UTF-8 without signature, codepage 650001 in Visual Studio
				glColor4f(1,1,1,1);
				
				sth_draw_text(stash, droidJapanese,32.f, dx, dy, (const char*) "\xE7\xA7\x81\xE3\x81\xAF\xE3\x82\xAC\xE3\x83\xA9\xE3\x82\xB9\xE3\x82\x92\xE9\xA3\x9F\xE3\x81\xB9\xE3\x82\x89\xE3\x82\x8C\xE3\x81\xBE\xE3\x81\x99\xE3\x80\x82",&dx);//はabcdefghijlkmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890!@#$%^&*()_-+=?/\][{}.,<>`~@#$%^", &dx);
//				sth_draw_text(stash, droidJapanese,32.f, dx, dy, (const char*) "私はガラスを食べられます。それは私を傷つけません。",&dx);//はabcdefghijlkmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890!@#$%^&*()_-+=?/\][{}.,<>`~@#$%^", &dx);
				
				dx = sx;

				sth_flush_draw(stash);
				glColor4f(0,0,0,1);
				//sth_draw_text(stash, droidRegular,32.f, dx-2, dy+2, "abcdefghijlkmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890!@#$%^", &dx);
				sth_flush_draw(stash);

			}	else
			{
				dx = sx;
				dy = height;
				glColor4f(1,1,1,1);

				sth_draw_texture(stash, droidRegular, 32.f, 0, 0,width,height, "a", &dx);
			}
			once++;
		}

		sth_end_draw(stash);
		
		glEnable(GL_DEPTH_TEST);
		//glFinish();
	}
#endif //_WIN32

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
				//printStats  = false;
			} else
			{
//				printf(".");
			}
		 }
		

	}

#ifdef _WIN32
	sth_delete(stash);
	free(data);
#endif

	render.CleanupShaders();
	window->exit();
	delete window;
	
	
	
	return 0;

#ifdef _WIN32
error_add_font:
	fprintf(stderr, "Could not add font.\n");

	render.CleanupShaders();
	window->exit();
	delete window;
	return -1;
#endif

}