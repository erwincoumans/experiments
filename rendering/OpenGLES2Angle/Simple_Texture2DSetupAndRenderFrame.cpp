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
//#include "esUtil.h"
#include "../NativeClient/shader_util.h"

#ifdef __APPLE__
#import <OpenGLES/EAGL.h>
#import <OpenGLES/ES1/gl.h>
#define	USE_IPHONE_SDK_JPEGLIB
#else
#include "GLES2/gl2.h"
#include "EGL/egl.h"

#endif//__APPLE__

#include <stdio.h>
#include "LinearMath/btTransform.h"
#include "btTransformUtil.h"
#include "OolongReadBlend.h"
#include "btBulletDynamicsCommon.h"

// Handle to a program object
GLuint programObject;
// Attribute locations
GLint  positionLoc;
GLint  texCoordLoc;
// Sampler location
GLint samplerLoc;

GLint modelMatrix;
GLint viewMatrix;
GLint projectionMatrix;

// Texture handle
GLuint textureId;

bool simulationPaused = false;


OolongBulletBlendReader* reader = 0;
btDiscreteDynamicsWorld* dynWorld = 0;
btDefaultCollisionConfiguration* collisionConfiguration = 0;
btCollisionDispatcher* dispatcher = 0;
btDbvtBroadphase* broadphase = 0;
btSequentialImpulseConstraintSolver* solver = 0;

void	zoomCamera(int deltaY)
{
	if (reader)
	{
		btVector3 fwd = reader->m_cameraTrans.getBasis().getColumn(2);
		btVector3 curPos = reader->m_cameraTrans.getOrigin();
		reader->m_cameraTrans.setOrigin(curPos+fwd*0.01f*(float)deltaY);
	}
}

//#define LOAD_FROM_FILE 1

#ifndef LOAD_FROM_FILE
#include "PhysicsAnimationBakingDemo.h"
#endif //LOAD_FROM_FILE

void createWorld()
{

#ifdef LOAD_FROM_FILE
	FILE* file = fopen("PhysicsAnimationBakingDemo2.blend","rb");
//FILE* file = fopen("p2p2.blend","rb");

	
	int fileLen=0;
	char*memoryBuffer =  0;

	{
		long currentpos = ftell(file); /* save current cursor position */
		long newpos;
		int bytesRead = 0;

		fseek(file, 0, SEEK_END); /* seek to end */
		newpos = ftell(file); /* find position of end -- this is the length */
		fseek(file, currentpos, SEEK_SET); /* restore previous cursor position */
		
		fileLen = newpos;
		
		memoryBuffer = (char*)malloc(fileLen);
		bytesRead = fread(memoryBuffer,fileLen,1,file);

		FILE* fileWrite = fopen("PhysicsAnimationBakingDemo.h","w");
		
		
		fprintf(fileWrite,"const char* mydata = {\n");
		int counter = 0;
		for (int i=0;i<fileLen;i++)
		{
			fprintf(fileWrite,"%d,",memoryBuffer[i]);
			counter++;
			if (counter>50)
			{
				counter=0;
				fprintf(fileWrite,"\n");
			}

		}
		fprintf(fileWrite,"};\n");
		fclose(fileWrite);

	}
		fclose(file);
#else
	int fileLen=sizeof(mydata);
	char*memoryBuffer =  mydata;

#endif //LOAD_FROM_FILE



	if (memoryBuffer && fileLen)
	{
			///collision configuration contains default setup for memory, collision setup
		collisionConfiguration = new btDefaultCollisionConfiguration();

		///use the default collision dispatcher. For parallel processing you can use a diffent dispatcher (see Extras/BulletMultiThreaded)
		dispatcher = new	btCollisionDispatcher(collisionConfiguration);

		broadphase = new btDbvtBroadphase();

		///the default constraint solver. For parallel processing you can use a different solver (see Extras/BulletMultiThreaded)
		solver = new btSequentialImpulseConstraintSolver;
	
		dynWorld = new btDiscreteDynamicsWorld(dispatcher,broadphase,solver,collisionConfiguration);

		reader = new OolongBulletBlendReader(dynWorld);
		int result = reader->readFile(memoryBuffer,fileLen);
		reader->convertAllObjects();
	}

}



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





///
// Create a simple 2x2 texture image with four different colors
//
GLuint CreateSimpleTexture2D( )
{
   // Texture object handle
   GLuint textureId;
   
   // 2x2 Image, 3 bytes per pixel (R, G, B)
   GLubyte pixels[4 * 3] =
   {  
      255,   0,   0, // Red
        0, 255,   0, // Green
        0,   0, 255, // Blue
      255, 255,   0  // Yellow
   };

   // Use tightly packed data
   glPixelStorei ( GL_UNPACK_ALIGNMENT, 1 );

   // Generate a texture object
   glGenTextures ( 1, &textureId );

   // Bind the texture object
   glBindTexture ( GL_TEXTURE_2D, textureId );

   // Load the texture
   glTexImage2D ( GL_TEXTURE_2D, 0, GL_RGB, 2, 2, 0, GL_RGB, GL_UNSIGNED_BYTE, pixels );

   // Set the filtering mode
   glTexParameteri ( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
   glTexParameteri ( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );

   return textureId;

}



float projMat[16];


///
// Initialize the shader and program object
//
bool setupGraphics(int screenWidth, int screenHeight) 
{

	 glViewport ( 0, 0, screenWidth,screenHeight );

	glEnable(GL_DEPTH_TEST);

    printGLString("Version", GL_VERSION);
    printGLString("Vendor", GL_VENDOR);
    printGLString("Renderer", GL_RENDERER);
    printGLString("Extensions", GL_EXTENSIONS);

   GLbyte vShaderStr[] =  
	  "uniform mat4 modelMatrix;\n"
	  "uniform mat4 viewMatrix;\n"
	  "uniform mat4 projectionMatrix;\n"
	  "attribute vec4 a_position;   \n"
      "attribute vec2 a_texCoord;   \n"
      "varying vec2 v_texCoord;     \n"
      "void main()                  \n"
      "{                            \n"
	  "   gl_Position = (projectionMatrix*viewMatrix*modelMatrix)*a_position; \n"
      "   v_texCoord = a_texCoord;  \n"
      "}                            \n";
   
   GLbyte fShaderStr[] =  
      "precision mediump float;                            \n"
      "varying vec2 v_texCoord;                            \n"
      "uniform sampler2D s_texture;                        \n"
      "void main()                                         \n"
      "{                                                   \n"
      "  gl_FragColor = texture2D( s_texture, v_texCoord );\n"
      "}                                                   \n";

// for wireframe, use white color
//	  "  gl_FragColor = vec4(1.0,1.0,1.0,1.0);\n"

   // Load the shaders and get a linked program object
   programObject = shader_util::CreateProgramFromVertexAndFragmentShaders((const char*)vShaderStr, (const char*)fShaderStr);
	   
	   //0;//esLoadProgram ((const char*)vShaderStr, (const char*)fShaderStr );
   
   // Get the attribute locations
   positionLoc = glGetAttribLocation ( programObject, "a_position" );
   texCoordLoc = glGetAttribLocation ( programObject, "a_texCoord" );
   
   // Get the sampler location
   samplerLoc = glGetUniformLocation ( programObject, "s_texture" );

   modelMatrix = glGetUniformLocation ( programObject, "modelMatrix" );
   viewMatrix = glGetUniformLocation ( programObject, "viewMatrix" );
   projectionMatrix = glGetUniformLocation ( programObject, "projectionMatrix" );

   float aspect;
	btVector3 extents;

	if (screenWidth > screenHeight) 
	{
		aspect = screenWidth / (float)screenHeight;
		extents.setValue(aspect * 1.0f, 1.0f,0);
	} else 
	{
		aspect = screenHeight / (float)screenWidth;
		extents.setValue(1.0f, aspect*1.f,0);
	}
	
	float m_frustumZNear=1;
	float m_frustumZFar=1000;

	
	btCreateFrustum(-aspect * m_frustumZNear, aspect * m_frustumZNear, -m_frustumZNear, m_frustumZNear, m_frustumZNear, m_frustumZFar,projMat);

   // Load the texture
   textureId = 0;//CreateSimpleTexture2D ();

   glClearColor ( 1.2f, 0.2f, 0.2f, 0.2f );

   createWorld();


   return true;
}

///
// Draw a triangle using the shader pair created in Init()
//
void renderFrame() 
{
	 
  glClearColor ( 0.6f, 0.6f, 0.6f, 0.f );

   GLfloat vVertices[] = { -0.5f,  0.5f, 0.0f,  // Position 0
                            0.0f,  0.0f,        // TexCoord 0 
                           -0.5f, -0.5f, 0.0f,  // Position 1
                            0.0f,  1.0f,        // TexCoord 1
                            0.5f, -0.5f, 0.0f,  // Position 2
                            1.0f,  1.0f,        // TexCoord 2
                            0.5f,  0.5f, 0.0f,  // Position 3
                            1.0f,  0.0f         // TexCoord 3
                         };
   GLushort indices[] = { 0, 1, 2, 2,1,0};//0, 2, 3 };
      
  
   // Clear the color buffer
   glClear ( GL_COLOR_BUFFER_BIT |GL_DEPTH_BUFFER_BIT);

   // Use the program object
   glUseProgram ( programObject );

   // Load the vertex position
   glVertexAttribPointer ( positionLoc, 3, GL_FLOAT, 
                           GL_FALSE, 5 * sizeof(GLfloat), vVertices );
   // Load the texture coordinate
   glVertexAttribPointer ( texCoordLoc, 2, GL_FLOAT,
                           GL_FALSE, 5 * sizeof(GLfloat), &vVertices[3] );

   glEnableVertexAttribArray ( positionLoc );
   glEnableVertexAttribArray ( texCoordLoc );

//   // Bind the texture
   glActiveTexture ( GL_TEXTURE0 );
   glBindTexture ( GL_TEXTURE_2D, textureId );

   // Set the sampler texture unit to 0
   glUniform1i ( samplerLoc, 0 );

 
   btVector3 m_cameraPosition(0,0,-2);
   btVector3 m_cameraTargetPosition(0,0,0);
   btVector3 m_cameraUp(0,1,0);//1,0);
   static float mat2[16];

#define USE_CAM_FROM_FILE 1
#ifdef USE_CAM_FROM_FILE
   if (reader)
   {
		reader->m_cameraTrans.inverse().getOpenGLMatrix(mat2);
   } else
#endif
   {
      btCreateLookAt(m_cameraPosition,m_cameraTargetPosition,m_cameraUp,mat2);
   }

   glUniformMatrix4fv(viewMatrix,1,GL_FALSE,mat2);

   	glUniformMatrix4fv(projectionMatrix,1,GL_FALSE,projMat);


	btTransform tr;
	tr.setIdentity();

	static float mat1[16];
	tr.getOpenGLMatrix(mat1);
	glUniformMatrix4fv(modelMatrix,1,GL_FALSE,mat1);
	glDrawElements ( GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, indices );

	tr.setOrigin(btVector3(0,1,0));

	tr.getOpenGLMatrix(mat1);
	glUniformMatrix4fv(modelMatrix,1,GL_FALSE,mat1);
	glDrawElements ( GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, indices );

	if (reader)
	{
		for (int i=0;i<reader->m_graphicsObjects.size();i++)
		{
			reader->m_graphicsObjects[i].render(positionLoc,texCoordLoc,samplerLoc,modelMatrix);
		}
		dynWorld->setGravity(btVector3(0,0,-1));//-1,0));
		
		if (!simulationPaused)
		{
			dynWorld->stepSimulation(0.016f);
			dynWorld->stepSimulation(0.016f);
		}
	}

}

///
// Cleanup
//
void	shutdownGraphics()
{


   // Delete texture object
   glDeleteTextures ( 1, &textureId );

   // Delete program object
   glDeleteProgram ( programObject );
}

