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
#ifdef __native_client__
#include "../NativeClient/shader_util.h"
#else
#include "esUtil.h"
#endif//

#include "../NativeClient/base64.h"
#include "../NativeClient/base64.cpp"

#ifdef __APPLE__
#import <OpenGLES/EAGL.h>
#import <OpenGLES/ES1/gl.h>
#define	USE_IPHONE_SDK_JPEGLIB
#else
#include "GLES2/gl2.h"
#include "EGL/egl.h"

#endif//__APPLE__

#include <string>
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
GLint g_lastError = 0;

// Texture handle
GLuint textureId;

bool simulationPaused = false;


OolongBulletBlendReader* reader = 0;
btDiscreteDynamicsWorld* dynWorld = 0;
btDefaultCollisionConfiguration* collisionConfiguration = 0;
btCollisionDispatcher* dispatcher = 0;
btDbvtBroadphase* broadphase = 0;
btSequentialImpulseConstraintSolver* solver = 0;


btVector3 m_cameraPosition(0,0,-2);
btVector3 m_cameraTargetPosition(0,0,0);
btVector3 m_cameraUp(0,1,0);//1,0);
btRigidBody* pickedBody = 0;
btTypedConstraint* m_pickConstraint = 0;
int m_glutScreenWidth = 0;
int m_glutScreenHeight = 0;
float m_cameraDistance = 0.f;



//////////////////////////////////////////////////////////

void	zoomCamera(int deltaY)
{
	if (reader)
	{
		btVector3 fwd = reader->m_cameraTrans.getBasis().getColumn(2);
		btVector3 curPos = reader->m_cameraTrans.getOrigin();
		reader->m_cameraTrans.setOrigin(curPos+fwd*0.01f*(float)deltaY);
	}
}


int m_mouseButtons=0;
int m_mouseOldX=0;
int m_mouseOldY=0;
bool m_ortho = false;
bool use6Dof = false;
btVector3 gOldPickingPos;
btVector3 gHitPos;
btScalar gOldPickingDist;

btScalar mousePickClamping = 30.f;


btVector3	getRayTo(int x,int y)
{
	if (m_ortho)
	{
		btScalar aspect;
		btVector3 extents;
		aspect = m_glutScreenWidth / (btScalar)m_glutScreenHeight;
		extents.setValue(aspect * 1.0f, 1.0f,0);
		
		extents *= m_cameraDistance;
		btVector3 lower = m_cameraTargetPosition - extents;
		btVector3 upper = m_cameraTargetPosition + extents;

		btScalar u = x / btScalar(m_glutScreenWidth);
		btScalar v = (m_glutScreenHeight - y) / btScalar(m_glutScreenHeight);
		
		btVector3	p(0,0,0);
		p.setValue((1.0f - u) * lower.getX() + u * upper.getX(),(1.0f - v) * lower.getY() + v * upper.getY(),m_cameraTargetPosition.getZ());
		return p;
	}

	float top = 1.f;
	float bottom = -1.f;
	float nearPlane = 1.f;
	float tanFov = (top-bottom)*0.5f / nearPlane;
	float fov = btScalar(2.0) * btAtan(tanFov);

	btVector3	rayFrom = m_cameraPosition;
	btVector3 rayForward = (m_cameraTargetPosition-m_cameraPosition);
	rayForward.normalize();
	float farPlane = 10000.f;
	rayForward*= farPlane;

	btVector3 rightOffset;
	btVector3 vertical = m_cameraUp;

	btVector3 hor;
	hor = rayForward.cross(vertical);
	hor.normalize();
	vertical = hor.cross(rayForward);
	vertical.normalize();

	float tanfov = tanf(0.5f*fov);


	hor *= 2.f * farPlane * tanfov;
	vertical *= 2.f * farPlane * tanfov;

	btScalar aspect;
	
	aspect = m_glutScreenWidth / (btScalar)m_glutScreenHeight;
	
	hor*=aspect;


	btVector3 rayToCenter = rayFrom + rayForward;
	btVector3 dHor = hor * 1.f/float(m_glutScreenWidth);
	btVector3 dVert = vertical * 1.f/float(m_glutScreenHeight);


	btVector3 rayTo = rayToCenter - 0.5f * hor + 0.5f * vertical;
	rayTo += btScalar(x) * dHor;
	rayTo -= btScalar(y) * dVert;
	return rayTo;
}

void removePickingConstraint()
{
	if (m_pickConstraint && dynWorld)
	{
		dynWorld->removeConstraint(m_pickConstraint);
		delete m_pickConstraint;
		//printf("removed constraint %i",gPickingConstraintId);
		m_pickConstraint = 0;
		pickedBody->forceActivationState(ACTIVE_TAG);
		pickedBody->setDeactivationTime( 0.f );
		pickedBody = 0;
	}
}


void mouseFunc(int button, int state, int x, int y)
{
	if (state == 0) 
	{
        m_mouseButtons |= 1<<button;
    } else
	{
        m_mouseButtons = 0;
    }

	m_mouseOldX = x;
    m_mouseOldY = y;


	//printf("button %i, state %i, x=%i,y=%i\n",button,state,x,y);
	//button 0, state 0 means left mouse down

	btVector3 rayTo = getRayTo(x,y);

	switch (button)
	{
	

	case 0:
		{
			if (state==0)
			{


				//add a point to point constraint for picking
				if (dynWorld)
				{
					
					btVector3 rayFrom;
					if (m_ortho)
					{
						rayFrom = rayTo;
						rayFrom.setZ(-100.f);
					} else
					{
						rayFrom = m_cameraPosition;
					}
					
					btCollisionWorld::ClosestRayResultCallback rayCallback(rayFrom,rayTo);
					dynWorld->rayTest(rayFrom,rayTo,rayCallback);
					if (rayCallback.hasHit())
					{


						btRigidBody* body = (btRigidBody*)btRigidBody::upcast(rayCallback.m_collisionObject);
						if (body)
						{
							//other exclusions?
							if (!(body->isStaticObject() || body->isKinematicObject()))
							{
								pickedBody = body;
								pickedBody->setActivationState(DISABLE_DEACTIVATION);


								btVector3 pickPos = rayCallback.m_hitPointWorld;
								//printf("pickPos=%f,%f,%f\n",pickPos.getX(),pickPos.getY(),pickPos.getZ());


								btVector3 localPivot = body->getCenterOfMassTransform().inverse() * pickPos;

								

								


								if (use6Dof)
								{
									btTransform tr;
									tr.setIdentity();
									tr.setOrigin(localPivot);
									btGeneric6DofConstraint* dof6 = new btGeneric6DofConstraint(*body, tr,false);
									dof6->setLinearLowerLimit(btVector3(0,0,0));
									dof6->setLinearUpperLimit(btVector3(0,0,0));
									dof6->setAngularLowerLimit(btVector3(0,0,0));
									dof6->setAngularUpperLimit(btVector3(0,0,0));

									dynWorld->addConstraint(dof6);
									m_pickConstraint = dof6;

									dof6->setParam(BT_CONSTRAINT_STOP_CFM,0.8,0);
									dof6->setParam(BT_CONSTRAINT_STOP_CFM,0.8,1);
									dof6->setParam(BT_CONSTRAINT_STOP_CFM,0.8,2);
									dof6->setParam(BT_CONSTRAINT_STOP_CFM,0.8,3);
									dof6->setParam(BT_CONSTRAINT_STOP_CFM,0.8,4);
									dof6->setParam(BT_CONSTRAINT_STOP_CFM,0.8,5);

									dof6->setParam(BT_CONSTRAINT_STOP_ERP,0.1,0);
									dof6->setParam(BT_CONSTRAINT_STOP_ERP,0.1,1);
									dof6->setParam(BT_CONSTRAINT_STOP_ERP,0.1,2);
									dof6->setParam(BT_CONSTRAINT_STOP_ERP,0.1,3);
									dof6->setParam(BT_CONSTRAINT_STOP_ERP,0.1,4);
									dof6->setParam(BT_CONSTRAINT_STOP_ERP,0.1,5);
								} else
								{
									btPoint2PointConstraint* p2p = new btPoint2PointConstraint(*body,localPivot);
									dynWorld->addConstraint(p2p);
									m_pickConstraint = p2p;
									p2p->m_setting.m_impulseClamp = mousePickClamping;
									//very weak constraint for picking
									p2p->m_setting.m_tau = 0.001f;
/*
									p2p->setParam(BT_CONSTRAINT_CFM,0.8,0);
									p2p->setParam(BT_CONSTRAINT_CFM,0.8,1);
									p2p->setParam(BT_CONSTRAINT_CFM,0.8,2);
									p2p->setParam(BT_CONSTRAINT_ERP,0.1,0);
									p2p->setParam(BT_CONSTRAINT_ERP,0.1,1);
									p2p->setParam(BT_CONSTRAINT_ERP,0.1,2);
									*/
									

								}
								use6Dof = !use6Dof;

								//save mouse position for dragging
								gOldPickingPos = rayTo;
								gHitPos = pickPos;

								gOldPickingDist  = (pickPos-rayFrom).length();
							}
						}
					}
				}

			} else
			{
				removePickingConstraint();
			}

			break;

		}
	default:
		{
		}
	}

}





void	mouseMotionFunc(int x,int y)
{

	if (m_pickConstraint)
	{
		//move the constraint pivot

		if (m_pickConstraint->getConstraintType() == D6_CONSTRAINT_TYPE)
		{
			btGeneric6DofConstraint* pickCon = static_cast<btGeneric6DofConstraint*>(m_pickConstraint);
			if (pickCon)
			{
				//keep it at the same picking distance

				btVector3 newRayTo = getRayTo(x,y);
				btVector3 rayFrom;
				btVector3 oldPivotInB = pickCon->getFrameOffsetA().getOrigin();

				btVector3 newPivotB;
				if (m_ortho)
				{
					newPivotB = oldPivotInB;
					newPivotB.setX(newRayTo.getX());
					newPivotB.setY(newRayTo.getY());
				} else
				{
					rayFrom = m_cameraPosition;
					btVector3 dir = newRayTo-rayFrom;
					dir.normalize();
					dir *= gOldPickingDist;

					newPivotB = rayFrom + dir;
				}
				pickCon->getFrameOffsetA().setOrigin(newPivotB);
			}

		} else
		{
			btPoint2PointConstraint* pickCon = static_cast<btPoint2PointConstraint*>(m_pickConstraint);
			if (pickCon)
			{
				//keep it at the same picking distance

				btVector3 newRayTo = getRayTo(x,y);
				btVector3 rayFrom;
				btVector3 oldPivotInB = pickCon->getPivotInB();
				btVector3 newPivotB;
				if (m_ortho)
				{
					newPivotB = oldPivotInB;
					newPivotB.setX(newRayTo.getX());
					newPivotB.setY(newRayTo.getY());
				} else
				{
					rayFrom = m_cameraPosition;
					btVector3 dir = newRayTo-rayFrom;
					dir.normalize();
					dir *= gOldPickingDist;

					newPivotB = rayFrom + dir;
				}
				pickCon->setPivotB(newPivotB);
			}
		}
	}

	float dx, dy;
    dx = btScalar(x) - m_mouseOldX;
    dy = btScalar(y) - m_mouseOldY;


	

	m_mouseOldX = x;
    m_mouseOldY = y;
	

}




//////////////////////////////////////////////////////////









//#define LOAD_FROM_FILE 1

#ifndef LOAD_FROM_FILE
//#include "PhysicsAnimationBakingDemo.h"
#include "PhysicsAnimationBakingDemo.uue64.h"
#endif //LOAD_FROM_FILE

char* theData = mydata_base64;

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

	if (!theData)
		return;
	
	std::string decoded = base64_decode(theData);

	int fileLen=decoded.size();
	char*memoryBuffer =  (char*)decoded.c_str();


#endif //LOAD_FROM_FILE


	if (dynWorld)
	{
		delete dynWorld;
		delete solver;
		delete dispatcher;
		delete broadphase;
		delete collisionConfiguration;
		delete reader;
	}

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
		if (result)
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
	m_glutScreenWidth = screenWidth;
	m_glutScreenHeight = screenHeight;


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

   modelMatrix = glGetUniformLocation ( programObject, "modelMatrix" );
   viewMatrix = glGetUniformLocation ( programObject, "viewMatrix" );
   projectionMatrix = glGetUniformLocation ( programObject, "projectionMatrix" );

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


 

   static float mat2[16];

#define USE_CAM_FROM_FILE 1
#ifdef USE_CAM_FROM_FILE
   if (reader && reader->isBlendFileOk())
   {
		//reader->m_cameraTrans.inverse().getOpenGLMatrix(mat2);

		btVector3 fwd = reader->m_cameraTrans.getBasis().getColumn(2);
		m_cameraPosition = reader->m_cameraTrans.getOrigin();
		m_cameraTargetPosition = m_cameraPosition - fwd; //why is this -?
		m_cameraUp = reader->m_cameraTrans.getBasis().getColumn(1);
		btCreateLookAt(m_cameraPosition,m_cameraTargetPosition,m_cameraUp,mat2);
   } else
#endif
   {
      btCreateLookAt(m_cameraPosition,m_cameraTargetPosition,m_cameraUp,mat2);
   }

   glUniformMatrix4fv(viewMatrix,1,GL_FALSE,mat2);


    float aspect;
	btVector3 extents;

	if (m_glutScreenWidth > m_glutScreenHeight) 
	{
		aspect = m_glutScreenWidth / (float)m_glutScreenHeight;
		extents.setValue(aspect * 1.0f, 1.0f,0);
	} else 
	{
		aspect = m_glutScreenHeight / (float)m_glutScreenWidth;
		extents.setValue(1.0f, aspect*1.f,0);
	}
	
	float m_frustumZNear=1;
	float m_frustumZFar=1000;

	
	btCreateFrustum(-aspect * m_frustumZNear, aspect * m_frustumZNear, -m_frustumZNear, m_frustumZNear, m_frustumZNear, m_frustumZFar,projMat);

  

   	glUniformMatrix4fv(projectionMatrix,1,GL_FALSE,projMat);




	if (reader && reader->isBlendFileOk())
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
	} else
	{
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
	}

	g_lastError = glGetError();
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


void mouseMove()
{

}