
#include "GLES2Renderer.h"

#ifdef __APPLE__
	#import <OpenGLES/EAGL.h>
	#import <OpenGLES/ES1/gl.h>
	#define	USE_IPHONE_SDK_JPEGLIB
#else
	#include "GLES2/gl2.h"
	#include "EGL/egl.h"
#endif//__APPLE__


#include "LinearMath/btTransform.h"
#include "btTransformUtil.h"

#include "BulletDynamics/Dynamics/btDiscreteDynamicsWorld.h"
#include "BulletDynamics/Dynamics/btRigidBody.h"
#include "LinearMath/btDefaultMotionState.h"
#include "LinearMath/btIDebugDraw.h"
#include "GLES2ShapeDrawer.h"

#include <stdlib.h>
#include <stdio.h>

// Handle to a program object
GLuint programObject;
// Attribute locations
GLint  positionLoc;
GLint  texCoordLoc;
// Sampler location
GLint samplerLoc;

GLint localScaling;
GLint modelMatrix;
GLint viewMatrix;
GLint projectionMatrix;
GLint g_lastError = 0;

// Texture handle
GLuint textureId;

btVector3 m_cameraPosition(0,12,-32);
btVector3 m_cameraTargetPosition(0,0,0);
btVector3 m_cameraUp(0,1,0);
float	m_ele=20;
float m_azi=0;
float m_zoomStepSize = 0.4f;
float m_cameraDistance=20;
GLES2Renderer::GLES2Renderer()
	:m_debugMode(0),
	m_shapeDrawer(0)
{
}
GLES2Renderer::~GLES2Renderer()
{
}



void GLES2Renderer::updateCamera()
{
	btScalar rele = m_ele * btScalar(0.01745329251994329547);// rads per deg
	btScalar razi = m_azi * btScalar(0.01745329251994329547);// rads per deg

	btQuaternion rot(m_cameraUp,razi);

	btVector3 eyePos(0,0,0);
	int m_forwardAxis = 2;
	eyePos[m_forwardAxis] = -m_cameraDistance;

	btVector3 forward(eyePos[0],eyePos[1],eyePos[2]);
	if (forward.length2() < SIMD_EPSILON)
	{
		forward.setValue(1.f,0.f,0.f);
	}
	btVector3 right = m_cameraUp.cross(forward);
	btQuaternion roll(right,-rele);

	eyePos = btMatrix3x3(rot) * btMatrix3x3(roll) * eyePos;

	m_cameraPosition[0] = eyePos.getX();
	m_cameraPosition[1] = eyePos.getY();
	m_cameraPosition[2] = eyePos.getZ();
	m_cameraPosition += m_cameraTargetPosition;

}

const float STEPSIZE = 5;

void GLES2Renderer::stepLeft() 
{ 
	m_azi -= STEPSIZE; if (m_azi < 0) m_azi += 360; updateCamera(); 
}
void GLES2Renderer::stepRight() 
{ 
	m_azi += STEPSIZE; if (m_azi >= 360) m_azi -= 360; updateCamera(); 
}
void GLES2Renderer::stepFront() 
{ 
	m_ele += STEPSIZE; if (m_ele >= 360) m_ele -= 360; updateCamera(); 
}
void GLES2Renderer::stepBack() 
{ 
	m_ele -= STEPSIZE; if (m_ele < 0) m_ele += 360; updateCamera(); 
}
void GLES2Renderer::zoomIn() 
{ 
	m_cameraDistance -= btScalar(m_zoomStepSize); updateCamera(); 
	if (m_cameraDistance < btScalar(0.1))
		m_cameraDistance = btScalar(0.1);

}
void GLES2Renderer::zoomOut() 
{ 
	m_cameraDistance += btScalar(m_zoomStepSize); updateCamera(); 

}

void GLES2Renderer::keyboardCallback(unsigned char key)
{


	switch (key) 
	{

	case 'l' : stepLeft(); break;
	case 'r' : stepRight(); break;
	case 'f' : stepFront(); break;
	case 'b' : stepBack(); break;
	case 'z' : zoomIn(); break;
	case 'x' : zoomOut(); break;


	case 'd' : 
		if (m_debugMode & btIDebugDraw::DBG_NoDeactivation)
			m_debugMode = m_debugMode & (~btIDebugDraw::DBG_NoDeactivation);
		else
			m_debugMode |= btIDebugDraw::DBG_NoDeactivation;
		if (m_debugMode & btIDebugDraw::DBG_NoDeactivation)
		{
			gDisableDeactivation = true;
		} else
		{
			gDisableDeactivation = false;
		}
		break;



	

	default:
		//        std::cout << "unused key : " << key << std::endl;
		break;
	}
	

}


//
///
/// \brief Load a shader, check for compile errors, print error messages to output log
/// \param type Type of shader (GL_VERTEX_SHADER or GL_FRAGMENT_SHADER)
/// \param shaderSrc Shader source string
/// \return A new shader object on success, 0 on failure
//
GLuint  esLoadShader ( GLenum type, const char *shaderSrc )
{
   GLuint shader;
   GLint compiled;
   
   // Create the shader object
   shader = glCreateShader ( type );

   if ( shader == 0 )
   	return 0;

   // Load the shader source
   glShaderSource ( shader, 1, &shaderSrc, NULL );
   
   // Compile the shader
   glCompileShader ( shader );

   // Check the compile status
   glGetShaderiv ( shader, GL_COMPILE_STATUS, &compiled );

   if ( !compiled ) 
   {
      GLint infoLen = 0;

      glGetShaderiv ( shader, GL_INFO_LOG_LENGTH, &infoLen );
      
      if ( infoLen > 1 )
      {
         char* infoLog = (char*)malloc (sizeof(char) * infoLen );

         glGetShaderInfoLog ( shader, infoLen, NULL, infoLog );
         printf ( "Error compiling shader:\n%s\n", infoLog );            
         
         free ( infoLog );
      }

      glDeleteShader ( shader );
      return 0;
   }

   return shader;

}


//
///
/// \brief Load a vertex and fragment shader, create a program object, link program.
//         Errors output to log.
/// \param vertShaderSrc Vertex shader source code
/// \param fragShaderSrc Fragment shader source code
/// \return A new program object linked with the vertex/fragment shader pair, 0 on failure
//
GLuint  esLoadProgram ( const char *vertShaderSrc, const char *fragShaderSrc )
{
   GLuint vertexShader;
   GLuint fragmentShader;
   GLuint programObject;
   GLint linked;

   // Load the vertex/fragment shaders
   vertexShader = esLoadShader ( GL_VERTEX_SHADER, vertShaderSrc );
   if ( vertexShader == 0 )
      return 0;

   fragmentShader = esLoadShader ( GL_FRAGMENT_SHADER, fragShaderSrc );
   if ( fragmentShader == 0 )
   {
      glDeleteShader( vertexShader );
      return 0;
   }

   // Create the program object
   programObject = glCreateProgram ( );
   
   if ( programObject == 0 )
      return 0;

   glAttachShader ( programObject, vertexShader );
   glAttachShader ( programObject, fragmentShader );

   // Link the program
   glLinkProgram ( programObject );

   // Check the link status
   glGetProgramiv ( programObject, GL_LINK_STATUS, &linked );

   if ( !linked ) 
   {
      GLint infoLen = 0;

      glGetProgramiv ( programObject, GL_INFO_LOG_LENGTH, &infoLen );
      
      if ( infoLen > 1 )
      {
         char* infoLog = (char*)malloc (sizeof(char) * infoLen );

         glGetProgramInfoLog ( programObject, infoLen, NULL, infoLog );
         printf ( "Error linking program:\n%s\n", infoLog );            
         
         free ( infoLog );
      }

      glDeleteProgram ( programObject );
      return 0;
   }

   // Free up no longer needed shader resources
   glDeleteShader ( vertexShader );
   glDeleteShader ( fragmentShader );

   return programObject;
}

static void printGLString(const char *name, GLenum s) {
    const char *v = (const char *) glGetString(s);
    printf("GL %s = %s\n", name, v);
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

void GLES2Renderer::init(int screenWidth, int screenHeight)
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
	  "uniform vec4 localScaling;\n"
	  "attribute vec4 a_position;   \n"
      "attribute vec2 a_texCoord;   \n"
      "varying vec2 v_texCoord;     \n"
      "void main()                  \n"
      "{                            \n"
	  "   gl_Position = (projectionMatrix*viewMatrix*modelMatrix)*(a_position*localScaling); \n"
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
#ifdef __native_client__
   programObject = shader_util::CreateProgramFromVertexAndFragmentShaders((const char*)vShaderStr, (const char*)fShaderStr);
#else
	programObject= esLoadProgram ((const char*)vShaderStr, (const char*)fShaderStr );
#endif

   // Get the attribute locations
   positionLoc = glGetAttribLocation ( programObject, "a_position" );
   texCoordLoc = glGetAttribLocation ( programObject, "a_texCoord" );
   
   // Get the sampler location
   samplerLoc = glGetUniformLocation ( programObject, "s_texture" );

   localScaling = glGetUniformLocation(programObject,"localScaling");
   modelMatrix = glGetUniformLocation ( programObject, "modelMatrix" );
   viewMatrix = glGetUniformLocation ( programObject, "viewMatrix" );
   projectionMatrix = glGetUniformLocation ( programObject, "projectionMatrix" );

   // Load the texture
   textureId = CreateSimpleTexture2D ();

   glClearColor ( 1.2f, 0.2f, 0.2f, 0.2f );
}



void	GLES2Renderer::renderscene(int modelMatrix, const btDiscreteDynamicsWorld* world)
{
	btScalar	m[16];
	btMatrix3x3	rot;rot.setIdentity();
	const int	numObjects=world->getNumCollisionObjects();
	btVector3 wireColor(1,0,0);
	for(int i=0;i<numObjects;i++)
	{
		btCollisionObject*	colObj=world->getCollisionObjectArray()[i];
		btRigidBody*		body=btRigidBody::upcast(colObj);
		if(body&&body->getMotionState())
		{
			btDefaultMotionState* myMotionState = (btDefaultMotionState*)body->getMotionState();
			myMotionState->m_graphicsWorldTrans.getOpenGLMatrix(m);
			rot=myMotionState->m_graphicsWorldTrans.getBasis();
		}
		else
		{
			colObj->getWorldTransform().getOpenGLMatrix(m);
			rot=colObj->getWorldTransform().getBasis();
		}
		btVector3 wireColor(1.f,1.0f,0.5f); //wants deactivation
		if(i&1) wireColor=btVector3(0.f,0.0f,1.f);
		///color differently for active, sleeping, wantsdeactivation states
		if (colObj->getActivationState() == 1) //active
		{
			if (i & 1)
			{
				wireColor += btVector3 (1.f,0.f,0.f);
			}
			else
			{			
				wireColor += btVector3 (.5f,0.f,0.f);
			}
		}
		if(colObj->getActivationState()==2) //ISLAND_SLEEPING
		{
			if(i&1)
			{
				wireColor += btVector3 (0.f,1.f, 0.f);
			}
			else
			{
				wireColor += btVector3 (0.f,0.5f,0.f);
			}
		}

		btVector3 aabbMin,aabbMax;
		world->getBroadphase()->getBroadphaseAabb(aabbMin,aabbMax);
		
		aabbMin-=btVector3(BT_LARGE_FLOAT,BT_LARGE_FLOAT,BT_LARGE_FLOAT);
		aabbMax+=btVector3(BT_LARGE_FLOAT,BT_LARGE_FLOAT,BT_LARGE_FLOAT);
//		printf("aabbMin=(%f,%f,%f)\n",aabbMin.getX(),aabbMin.getY(),aabbMin.getZ());
//		printf("aabbMax=(%f,%f,%f)\n",aabbMax.getX(),aabbMax.getY(),aabbMax.getZ());
//		m_dynamicsWorld->getDebugDrawer()->drawAabb(aabbMin,aabbMax,btVector3(1,1,1));

		if (!(getDebugMode()& btIDebugDraw::DBG_DrawWireframe))
		{
			m_shapeDrawer->drawOpenGL(modelMatrix,positionLoc, texCoordLoc,localScaling, m,colObj->getCollisionShape(),wireColor,getDebugMode(),aabbMin,aabbMax);
//			
			//switch(pass)
			//{
			//case	0:	m_shapeDrawer->drawOpenGL(m,colObj->getCollisionShape(),wireColor,getDebugMode(),aabbMin,aabbMax);break;
//			case	1:	m_shapeDrawer->drawShadow(m,m_sundirection*rot,colObj->getCollisionShape(),aabbMin,aabbMax);break;
	//		case	2:	m_shapeDrawer->drawOpenGL(m,colObj->getCollisionShape(),wireColor*btScalar(0.3),0,aabbMin,aabbMax);break;
//			}
		}
	}
}


void GLES2Renderer::draw(const btDiscreteDynamicsWorld* world, int width, int height)
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


	static float projMat[16];

   static float mat2[16];
   
   btCreateLookAt(m_cameraPosition,m_cameraTargetPosition,m_cameraUp,mat2);

   glUniformMatrix4fv(viewMatrix,1,GL_FALSE,mat2);


    float aspect;
	btVector3 extents;

	if (width > height) 
	{
		aspect = width/ (float)height;
		extents.setValue(aspect * 1.0f, 1.0f,0);
	} else 
	{
		aspect = height / (float)width;
		extents.setValue(1.0f, aspect*1.f,0);
	}
	
	float m_frustumZNear=1;
	float m_frustumZFar=1000;

	
	btCreateFrustum(-aspect * m_frustumZNear, aspect * m_frustumZNear, -m_frustumZNear, m_frustumZNear, m_frustumZNear, m_frustumZFar,projMat);

  

   	glUniformMatrix4fv(projectionMatrix,1,GL_FALSE,projMat);
	glUniform4f(localScaling,1,1,1,1);

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

	renderscene(modelMatrix,world);
}