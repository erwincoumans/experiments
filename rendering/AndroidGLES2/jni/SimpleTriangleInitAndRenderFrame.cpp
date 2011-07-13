
// Simple_Texture2D.c
//
//    This is a simple example that draws a quad with a 2D
//    texture image. The purpose of this example is to demonstrate 
//    the basics of 2D texturing
//
#include <stdlib.h>
#include "esUtil.h"
#include <stdio.h>

// Handle to a program object
GLuint programObject;
// Attribute locations
GLint  positionLoc;
GLint  texCoordLoc;
// Sampler location
GLint samplerLoc;
// Texture handle
GLuint textureId;




static void printGLString(const char *name, GLenum s) {
    const char *v = (const char *) glGetString(s);
    printf("GL %s = %s\n", name, v);
}

static void checkGlError(const char* op) {
    for (GLint error = glGetError(); error; error
            = glGetError()) {
        printf("after %s() glError (0x%x)\n", op, error);
    }
}

static const char gVertexShader[] = 
    "attribute vec4 vPosition;\n"
    "void main() {\n"
    "  gl_Position = vPosition;\n"
    "}\n";

static const char gFragmentShader[] = 
    "precision mediump float;\n"
    "void main() {\n"
    "  gl_FragColor = vec4(0.0, 1.0, 0.0, 1.0);\n"
    "}\n";




bool setupGraphics(int w, int h) {
    printGLString("Version", GL_VERSION);
    printGLString("Vendor", GL_VENDOR);
    printGLString("Renderer", GL_RENDERER);
    printGLString("Extensions", GL_EXTENSIONS);

    printf("setupGraphics(%d, %d)", w, h);
    programObject = esLoadProgram(gVertexShader, gFragmentShader);
    if (!programObject) {
        printf("Could not create program.");
        return false;
    }
    positionLoc = glGetAttribLocation(programObject, "vPosition");
    checkGlError("glGetAttribLocation");
    printf("glGetAttribLocation(\"vPosition\") = %d\n",
            positionLoc);

    glViewport(0, 0, w, h);
    checkGlError("glViewport");
    return true;
}

const GLfloat gTriangleVertices[] = 
{ 0.0f, 0.5f, -0.5f, 
-0.5f,  0.5f, -0.5f 
};

void renderFrame() 
{
    static float grey;
    grey += 0.01f;
    if (grey > 1.0f) {
        grey = 0.0f;
    }
    glClearColor(grey, grey, grey, 1.0f);
    checkGlError("glClearColor");
    glClear( GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
    checkGlError("glClear");

    glUseProgram(programObject);
    checkGlError("glUseProgram");

    glVertexAttribPointer(positionLoc, 2, GL_FLOAT, GL_FALSE, 0, gTriangleVertices);
    checkGlError("glVertexAttribPointer");
    glEnableVertexAttribArray(positionLoc);
    checkGlError("glEnableVertexAttribArray");
    glDrawArrays(GL_TRIANGLES, 0, 3);
    checkGlError("glDrawArrays");
}

void	shutdownGraphics()
{
   // Delete texture object
   glDeleteTextures ( 1, &textureId );

   // Delete program object
   glDeleteProgram ( programObject );
}