//
// Book:      OpenGL(R) ES 2.0 Programming Guide
// Authors:   Aaftab Munshi, Dan Ginsburg, Dave Shreiner
// ISBN-10:   0321502795
// ISBN-13:   9780321502797
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/9780321563835
//            http://www.opengles-book.com
//

// Simple_Texture2D.c
//
//    This is a simple example that draws a quad with a 2D
//    texture image. The purpose of this example is to demonstrate 
//    the basics of 2D texturing
//
#include <stdlib.h>
#include "esUtil.h"
#include <stdio.h>

extern bool setupGraphics(int w, int h);
extern void renderFrame() ;
extern void shutdownGraphics();

static void printGLString(const char *name, GLenum s) {
    const char *v = (const char *) glGetString(s);
    printf("GL %s = %s\n", name, v);
}

///
// Initialize the shader and program object
//
int Init (ESContext *esContext )
{

   setupGraphics(esContext->width,esContext->height);

   glClearColor ( 0.0f, 0.0f, 0.0f, 0.0f );
   return TRUE;
}

///
// Draw a triangle using the shader pair created in Init()
//
void Draw ( ESContext *esContext )
{
	 // Set the viewport
   glViewport ( 0, 0, esContext->width, esContext->height );
   
   renderFrame();

   eglSwapBuffers ( esContext->eglDisplay, esContext->eglSurface );
}

///
// Cleanup
//
void ShutDown ( ESContext *esContext )
{
	shutdownGraphics();

}



int main ( int argc, char *argv[] )
{
   ESContext esContext;


   esInitContext ( &esContext );
   esContext.userData = 0;

//   esCreateWindow ( &esContext, TEXT("Simple Texture 2D"), 320, 240, ES_WINDOW_RGB );
     esCreateWindow ( &esContext, TEXT("Simple Texture 2D"), 640, 480, ES_WINDOW_RGB|ES_WINDOW_DEPTH|ES_WINDOW_ALPHA );
   
   if ( !Init (&esContext) )
      return 0;

    printGLString("Version", GL_VERSION);
    printGLString("Vendor", GL_VENDOR);
    printGLString("Renderer", GL_RENDERER);
    printGLString("Extensions", GL_EXTENSIONS);

   esRegisterDrawFunc ( &esContext, Draw );
   
   esMainLoop ( &esContext );

   ShutDown ( &esContext );
}