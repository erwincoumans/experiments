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

#include "GLInstancingRenderer.h"

#ifdef __APPLE__
#include "MacOpenGLWindow.h"
#else
#include "Win32OpenGLRenderManager.h"
#endif

#include "GLPrimitiveRenderer.h"


#include "RenderScene.h"


#include "LinearMath/btQuickprof.h"
#include "LinearMath/btQuaternion.h"

#include "../../opencl/gpu_rigidbody_pipeline/CommandlineArgs.h"

#include "../OpenGLTrueTypeFont/fontstash.h"

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


#include "gwenWindow.h"


void MyMouseCallback(int button, int state, float x, float y)
{
	btDefaultMouseCallback(button,state,x,y);

	static int m_lastmousepos[2] = {0,0};
	static bool isInitialized = false;
	if (pCanvas)
	{
		if (!isInitialized)
		{
			isInitialized = true;
			m_lastmousepos[0] = x+1;
			m_lastmousepos[1] = y+1;
		}
		bool handled = pCanvas->InputMouseMoved(x,y,m_lastmousepos[0],m_lastmousepos[1]);

		if (button>=0)
		{
			handled = pCanvas->InputMouseButton(button,state);
			if (handled)
			{
				m_lastmousepos[0]	=	x;
				m_lastmousepos[1]	=	y;
			
				if (!state)
					return;
			}
		}
		
	}
}

	int droidRegular, droidItalic, droidBold, droidJapanese, dejavu;

sth_stash* initFont()
{
	GLint err;

		struct sth_stash* stash = 0;
	FILE* fp = 0;
	int datasize;
	unsigned char* data;
	float sx,sy,dx,dy,lh;
	GLuint texture;

	stash = sth_create(512,512);//256,256);//,1024);//512,512);
    err = glGetError();
    assert(err==GL_NO_ERROR);
    
	if (!stash)
	{
		fprintf(stderr, "Could not create stash.\n");
		return 0;
	}

	// Load the first truetype font from memory (just because we can).
#ifdef _WIN32
    const char* fontPath = "../../bin/";
#else
    const char* fontPath = "./";
#endif
    
    char fullFontFileName[1024];
    sprintf(fullFontFileName,"%s%s",fontPath,"DroidSerif-Regular.ttf");//cour.ttf");//times.ttf");//DroidSerif-Regular.ttf");
    
	fp = fopen(fullFontFileName, "rb");

    err = glGetError();
    assert(err==GL_NO_ERROR);
    
    assert(fp);
    if (fp)
    {
        fseek(fp, 0, SEEK_END);
        datasize = (int)ftell(fp);
        fseek(fp, 0, SEEK_SET);
        data = (unsigned char*)malloc(datasize);
        if (data == NULL)
        {
            assert(0);
            return 0;
        }
        else
            fread(data, 1, datasize, fp);
        fclose(fp);
        fp = 0;
    }
	if (!(droidRegular = sth_add_font_from_memory(stash, data)))
    {
        assert(0);
        return 0;
    }
    err = glGetError();
    assert(err==GL_NO_ERROR);

	// Load the remaining truetype fonts directly.
    sprintf(fullFontFileName,"%s%s",fontPath,"DroidSerif-Italic.ttf");

	if (!(droidItalic = sth_add_font(stash,fullFontFileName)))
	{
        assert(0);
        return 0;
    }
     sprintf(fullFontFileName,"%s%s",fontPath,"DroidSerif-Bold.ttf");

	if (!(droidBold = sth_add_font(stash,fullFontFileName)))
	{
        assert(0);
        return 0;
    }
    err = glGetError();
    assert(err==GL_NO_ERROR);
    
     sprintf(fullFontFileName,"%s%s",fontPath,"DroidSansJapanese.ttf");
    if (!(droidJapanese = sth_add_font(stash,fullFontFileName)))
	{
        assert(0);
        return 0;
    }
    err = glGetError();
    assert(err==GL_NO_ERROR);

	return stash;
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

	
	sth_stash* stash = initFont();
		
	render.InitShaders();


	createSceneProgrammatically(render);
    

	render.writeTransforms();

    window->runMainLoop();

	window->setMouseCallback(MyMouseCallback);
	window->setKeyboardCallback(btDefaultKeyboardCallback);


    GLPrimitiveRenderer prender(width,height);
   
   glUseProgram(0); 

////////////////////////////////
	setupGUI(width,height);

	/////////////////////////////////////


	if (pCanvas)
	{
		pCanvas->SetSize(width,height);
	}

	class CProfileIterator* m_profileIterator;
	m_profileIterator = CProfileManager::Get_Iterator();
	processProfileData(m_profileIterator,false);

	glClearColor(1,1,1,1);
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
			//render.writeSingleInstanceTransformToGPU(pos,orn,0);
//			createScene(render, demo);
//			printf("numPhysicsInstances= %d\n", demo.m_numPhysicsInstances);
//			printf("numDynamicPhysicsInstances= %d\n", demo.m_numDynamicPhysicsInstances);
//			render.writeTransforms();
		}



		window->startRendering();
		render.RenderScene();
		float col[4]={0,1,0,1};
		prender.drawRect(10,50,120,60,col);
//             glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
		glEnable(GL_TEXTURE_2D);
    
			glEnable(GL_BLEND);
        err = glGetError();
        assert(err==GL_NO_ERROR);

		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		
        err = glGetError();
        assert(err==GL_NO_ERROR);

        err = glGetError();
        assert(err==GL_NO_ERROR);
        
		glDisable(GL_DEPTH_TEST);
        err = glGetError();
        assert(err==GL_NO_ERROR);
        
		//glColor4ub(255,0,0,255);
		
        err = glGetError();
        assert(err==GL_NO_ERROR);
        
		
        err = glGetError();
        assert(err==GL_NO_ERROR);
        glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
    

		float x = 10;
		float y=100;
		float  dx=0;
		sth_begin_draw(stash);
		glColor4f(1,1,1,1);
		sth_flush_draw(stash);
		sth_draw_text(stash, droidRegular,40.f, x, y, "How does this OpenGL True Type font look? ", &dx,width,height);
		sth_flush_draw(stash);
		
		sth_end_draw(stash);


		if (pCanvas)
		{
		//	saveOpenGLState(width,height);//m_glutScreenWidth,m_glutScreenHeight);
			glMatrixMode(GL_TEXTURE);
			glLoadIdentity();
			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
			glMatrixMode(GL_PROJECTION);
			glLoadIdentity();
			glViewport(0,0,width-1,height-1);
			glOrtho(0, width-1, height-1, 0, -1, 1);
			glDisable(GL_TEXTURE_GEN_S);
			glDisable(GL_TEXTURE_GEN_T);
			glDisable(GL_TEXTURE_GEN_R);
			glDisable(GL_CULL_FACE);
			glDisable(GL_DEPTH_TEST);
			glDisable(GL_LIGHTING);
			glEnable(GL_BLEND);

			pCanvas->RenderCanvas();
			//restoreOpenGLState();
		}


		window->endRendering();


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

//	render.CleanupShaders();
	window->exit();
	delete window;
	
	
	
	return 0;


}
