

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


#include "GLES2AngleWindow.h"

#include <windows.h>
#include "LinearMath/btVector3.h"

#include <stdio.h>
#include <stdlib.h>
#include "GLES2/gl2.h"
#include "EGL/egl.h"

#include "../../rendering/rendertest/Win32InternalWindowData.h"

struct GLES2Context
{
	EGLDisplay m_display;
	EGLContext m_context;
	EGLSurface m_surface;
};

bool GLES2AngleWindow::enableGLES2()
{

	m_esContext = new GLES2Context;
	   EGLint attribList[] =
   {
       EGL_RED_SIZE,       5,
       EGL_GREEN_SIZE,     6,
       EGL_BLUE_SIZE,      5,
       EGL_ALPHA_SIZE,     8,// : EGL_DONT_CARE,
       EGL_DEPTH_SIZE,     8,// : EGL_DONT_CARE,
       EGL_STENCIL_SIZE,   8,// : EGL_DONT_CARE,
       EGL_SAMPLE_BUFFERS, 0,//(flags & ES_WINDOW_MULTISAMPLE) ? 1 : 0,
       EGL_NONE
   };

	 EGLint numConfigs;
   EGLint majorVersion;
   EGLint minorVersion;
   EGLDisplay display;
   EGLContext context;
   EGLSurface surface;
   EGLConfig config;
   EGLint contextAttribs[] = { EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE, EGL_NONE };

   // Get Display
   display = eglGetDisplay(GetDC(m_data->m_hWnd));
   if ( display == EGL_NO_DISPLAY )
   {
      return EGL_FALSE;
   }

   // Initialize EGL
   if ( !eglInitialize(display, &majorVersion, &minorVersion) )
   {
      return EGL_FALSE;
   }

   // Get configs
   if ( !eglGetConfigs(display, NULL, 0, &numConfigs) )
   {
      return EGL_FALSE;
   }

   // Choose config
   if ( !eglChooseConfig(display, attribList, &config, 1, &numConfigs) )
   {
      return EGL_FALSE;
   }

   // Create a surface
   surface = eglCreateWindowSurface(display, config, (EGLNativeWindowType)m_data->m_hWnd, NULL);
   if ( surface == EGL_NO_SURFACE )
   {
      return EGL_FALSE;
   }

   // Create a GL context
   context = eglCreateContext(display, config, EGL_NO_CONTEXT, contextAttribs );
   if ( context == EGL_NO_CONTEXT )
   {
      return EGL_FALSE;
   }   
   
   // Make the context current
   if ( !eglMakeCurrent(display, surface, surface, context) )
   {
      return EGL_FALSE;
   }
   
   m_esContext->m_display = display;
   m_esContext->m_surface = surface;
   m_esContext->m_context = context;
   return EGL_TRUE;

}



void GLES2AngleWindow::disableGLES2()
{
	eglMakeCurrent(m_esContext->m_display,m_esContext->m_surface,m_esContext->m_surface,0);
	eglDestroyContext(m_esContext->m_display,m_esContext->m_context);
	eglDestroySurface(m_esContext->m_display,m_esContext->m_surface);
	eglTerminate(m_esContext->m_display);
}





void	GLES2AngleWindow::createWindow(const btgWindowConstructionInfo& ci)
{
	Win32Window::createWindow(ci);

	//VideoDriver = video::createOpenGLDriver(CreationParams, FileSystem, this);
	enableGLES2();

}




GLES2AngleWindow::GLES2AngleWindow()
	:m_OpenGLInitialized(false),
	m_esContext(0)
{
	
	
}

GLES2AngleWindow::~GLES2AngleWindow()
{
	if (m_OpenGLInitialized)
	{
			disableGLES2();
	}
	delete m_esContext;
}


void	GLES2AngleWindow::closeWindow()
{
	disableGLES2();

	Win32Window::closeWindow();
}


void	GLES2AngleWindow::startRendering()
{
	pumpMessage();

	glViewport ( 0, 0, m_data->m_openglViewportWidth, m_data->m_openglViewportHeight);

//		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);	//clear buffers
		
		//glCullFace(GL_BACK);
		//glFrontFace(GL_CCW);
	//	glEnable(GL_DEPTH_TEST);



}


void	GLES2AngleWindow::renderAllObjects()
{
}

void	GLES2AngleWindow::endRendering()
{
#if 0 
	SwapBuffers( m_data->m_hDC );
#endif
	
	eglSwapBuffers ( m_esContext->m_display, m_esContext->m_surface);

	ValidateRect( m_data->m_hWnd, NULL );
}



	
