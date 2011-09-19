// Copyright (c) 2011 The Native Client Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
#define NOMINMAX 1
#include "cube.h"

#include <algorithm>
#include <vector>

#include "shader_util.h"
#include "transforms.h"


#ifdef __APPLE__
#import <OpenGLES/EAGL.h>
#import <OpenGLES/ES1/gl.h>
#define	USE_IPHONE_SDK_JPEGLIB
#else
#include "GLES2/gl2.h"
#include "EGL/egl.h"

#endif//__APPLE__

//?#include "OolongReadBlend.h"
#include "btBulletDynamicsCommon.h"
#include "btTransformUtil.h"

//#define SWAP_COORDINATE_SYSTEMS
#ifdef SWAP_COORDINATE_SYSTEMS

#define IRR_X 0
#define IRR_Y 2
#define IRR_Z 1

#define IRR_X_M 1.f
#define IRR_Y_M 1.f
#define IRR_Z_M 1.f

///also winding is different
#define IRR_TRI_0_X 0
#define IRR_TRI_0_Y 2
#define IRR_TRI_0_Z 1

#define IRR_TRI_1_X 0
#define IRR_TRI_1_Y 3
#define IRR_TRI_1_Z 2
#else
#define IRR_X 0
#define IRR_Y 1
#define IRR_Z 2

#define IRR_X_M 1.f
#define IRR_Y_M 1.f
#define IRR_Z_M 1.f

///also winding is different
#define IRR_TRI_0_X 0
#define IRR_TRI_0_Y 1
#define IRR_TRI_0_Z 2

#define IRR_TRI_1_X 0
#define IRR_TRI_1_Y 2
#define IRR_TRI_1_Z 3
#endif

// Handle to a program object
GLuint programObject;
// Attribute locations
GLint  positionLoc;
GLint  normalLoc;

GLint  texCoordLoc;
// Sampler location
GLint samplerLoc;

GLint modelMatrix;
GLint viewMatrix;
GLint projectionMatrix;

// Texture handle
GLuint textureId;

static void printGLString(const char *name, GLenum s) {
    const char *v = (const char *) glGetString(s);
    printf("GL %s = %s\n", name, v);
}

static void checkGlError(const char* op) {
    for (GLint error = glGetError(); error; error = glGetError()) 
	{
        printf("after %s() glError (0x%x)\n", op, error);
    }
}


#define NOR_SHORTTOFLOAT 32767.0f
void norShortToFloat(const short *shnor, float *fnor)
{
	fnor[0] = shnor[0] / NOR_SHORTTOFLOAT;
	fnor[1] = shnor[1] / NOR_SHORTTOFLOAT;
	fnor[2] = shnor[2] / NOR_SHORTTOFLOAT;
}




struct BasicTexture
{
	unsigned char*	m_jpgData;
	int		m_jpgSize;
	
	int				m_width;
	int				m_height;
	GLuint			m_textureName;
	bool			m_initialized;
	GLint			m_pixelColorComponents;
	
	//contains the uncompressed R8G8B8 pixel data
	unsigned char*	m_output;

	BasicTexture(unsigned char* textureData,int width,int height)
	:m_jpgData(0),
	m_jpgSize(0),
	m_width(width),
	m_height(height),
	m_output(textureData),
	m_initialized(false)
	{
		m_pixelColorComponents = GL_RGB;
	}		
	
	BasicTexture(unsigned char* jpgData,int jpgSize)
	: m_jpgData(jpgData),
	m_jpgSize(jpgSize),
	m_output(0),
	m_textureName(-1),
	m_initialized(false)
	{
		m_pixelColorComponents = GL_RGBA;
	}
	
	virtual ~BasicTexture()
	{
		delete[] m_output;
	}
	
	//returns true if szFilename has the szExt extension
	bool checkExt(char const * szFilename, char const * szExt)
	{
		if (strlen(szFilename) > strlen(szExt))
		{
			char const * szExtension = &szFilename[strlen(szFilename) - strlen(szExt)];
			if (!strcmp(szExtension, szExt))
				return true;
		}
		return false;
	}
	
	void	loadTextureMemory(const char* fileName)
	{
		if (checkExt(fileName,".JPG") || checkExt(fileName,".jpg"))
		{
			loadJpgMemory();
		}
	}
	
	void	initOpenGLTexture()
	{
		if (m_initialized)
		{
			glBindTexture(GL_TEXTURE_2D,m_textureName);
		} else
		{
			m_initialized = true;
			

			glGenTextures(1, &m_textureName);
			glBindTexture(GL_TEXTURE_2D,m_textureName);

			//glPixelStorei ( GL_UNPACK_ALIGNMENT, 1 );

//			gluBuild2DMipmaps(GL_TEXTURE_2D,3,m_width,m_height,GL_RGB,GL_UNSIGNED_BYTE,m_output);
//			glTexImage2D(GL_TEXTURE_2D,0,m_pixelColorComponents,m_width,m_height,0,m_pixelColorComponents,GL_UNSIGNED_BYTE,m_output);
			glTexImage2D(GL_TEXTURE_2D,0,GL_RGB,m_width,m_height,0,GL_RGB,GL_UNSIGNED_BYTE,m_output);
			
			glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			
//			glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR_MIPMAP_LINEAR);
	//		glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR_MIPMAP_LINEAR);
			glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_REPEAT);
			glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_REPEAT);

		}
		checkGlError("bla");
	}
	
	void	loadJpgMemory()
	{
		
#ifdef  USE_IPHONE_SDK_JPEGLIB
		NSData *imageData = [NSData dataWithBytes:m_jpgData length:m_jpgSize];
//		NSData *imageData = [NSData dataWithBytesNoCopy:m_jpgData length:m_jpgSize freeWhenDone:NO];
		UIImage *uiImage = [UIImage imageWithData:imageData];
		
		CGImageRef textureImage;
		CGContextRef textureContext;

		if( uiImage ) {
			textureImage = uiImage.CGImage;
			
			m_width = CGImageGetWidth(textureImage);
			m_height = CGImageGetHeight(textureImage);
			
						
			if(textureImage) {
				m_output = (GLubyte *) malloc(m_width * m_height * 4);
				textureContext = CGBitmapContextCreate(m_output, m_width, m_height, 8, m_width * 4, CGImageGetColorSpace(textureImage), kCGImageAlphaPremultipliedLast);
				CGContextDrawImage(textureContext, CGRectMake(0.0, 0.0, (float)m_width, (float)m_height), textureImage);
			}
			
		}			

#else
#ifdef USE_JPEG
		unsigned char **rowPtr=0;

		// allocate and initialize JPEG decompression object
		struct jpeg_decompress_struct cinfo;
		struct my_jpeg_error_mgr jerr;

		//We have to set up the error handler first, in case the initialization
		//step fails.  (Unlikely, but it could happen if you are out of memory.)
		//This routine fills in the contents of struct jerr, and returns jerr's
		//address which we place into the link field in cinfo.

		cinfo.err = jpeg_std_error(&jerr.pub);
		cinfo.err->error_exit = error_exit;
		cinfo.err->output_message = output_message;

		// compatibility fudge:
		// we need to use setjmp/longjmp for error handling as gcc-linux
		// crashes when throwing within external c code
		if (setjmp(jerr.setjmp_buffer))
		{
			// If we get here, the JPEG code has signaled an error.
			// We need to clean up the JPEG object and return.

			jpeg_destroy_decompress(&cinfo);

			
			// if the row pointer was created, we delete it.
			if (rowPtr)
				delete [] rowPtr;

			// return null pointer
			return ;
		}

		// Now we can initialize the JPEG decompression object.
		jpeg_create_decompress(&cinfo);

		// specify data source
		jpeg_source_mgr jsrc;

		// Set up data pointer
		jsrc.bytes_in_buffer = m_jpgSize;
		jsrc.next_input_byte = (JOCTET*)m_jpgData;
		cinfo.src = &jsrc;

		jsrc.init_source = init_source;
		jsrc.fill_input_buffer = fill_input_buffer;
		jsrc.skip_input_data = skip_input_data;
		jsrc.resync_to_restart = jpeg_resync_to_restart;
		jsrc.term_source = term_source;

		// Decodes JPG input from whatever source
		// Does everything AFTER jpeg_create_decompress
		// and BEFORE jpeg_destroy_decompress
		// Caller is responsible for arranging these + setting up cinfo

		// read file parameters with jpeg_read_header()
		jpeg_read_header(&cinfo, TRUE);

		cinfo.out_color_space=JCS_RGB;
		cinfo.out_color_components=3;
		cinfo.do_fancy_upsampling=FALSE;

		// Start decompressor
		jpeg_start_decompress(&cinfo);

		// Get image data
		unsigned short rowspan = cinfo.image_width * cinfo.out_color_components;
		m_width = cinfo.image_width;
		m_height = cinfo.image_height;

		// Allocate memory for buffer
		m_output = new unsigned char[rowspan * m_height];

		// Here we use the library's state variable cinfo.output_scanline as the
		// loop counter, so that we don't have to keep track ourselves.
		// Create array of row pointers for lib
		rowPtr = new unsigned char* [m_height];

		for( unsigned int i = 0; i < m_height; i++ )
			rowPtr[i] = &m_output[ i * rowspan ];

		unsigned int rowsRead = 0;

		while( cinfo.output_scanline < cinfo.output_height )
			rowsRead += jpeg_read_scanlines( &cinfo, &rowPtr[rowsRead], cinfo.output_height - rowsRead );

		delete [] rowPtr;
		// Finish decompression

		jpeg_finish_decompress(&cinfo);

		// Release JPEG decompression object
		// This is an important step since it will release a good deal of memory.
		jpeg_destroy_decompress(&cinfo);
#else
	m_width  = 2;
	m_height = 2;
	m_output = new unsigned char[m_width*4 * m_height];
	for (int i=0;i<m_width*4 * m_height;i++)
	{
		m_output[i] = 255;
	}
	m_output[0] = 0;

#endif //USE_JPEG
#endif
		
	}
	
};


GfxObject::GfxObject(GLuint vboId,btCollisionObject* colObj)
:m_colObj(colObj),
m_texture(0)
{
}


void GfxObject::render(int positionLoc,int normalLoc, int texCoordLoc, int samplerLoc, int modelMatrix)
{
	glFrontFace(GL_CCW);
//	glCullFace(GL_BACK);
	glCullFace(GL_FRONT);
	
	glDisable(GL_CULL_FACE);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc( GL_LEQUAL );
	if (0)//m_texture)
	{
//		glEnable(GL_TEXTURE_2D);
		glActiveTexture ( GL_TEXTURE0 );

		m_texture->initOpenGLTexture();


		glBindTexture(GL_TEXTURE_2D,m_texture->m_textureName);
		
//		glDisable(GL_TEXTURE_GEN_S);
	//	glDisable(GL_TEXTURE_GEN_T);
		//glDisable(GL_TEXTURE_GEN_R);
		
	} else
	{
	}


#ifdef USE_OPENGLES_1
	dsda
	glPushMatrix();
	float m[16];
	m_colObj->getWorldTransform().getOpenGLMatrix(m);
	
	glMultMatrixf(m);
		
//	glScalef(0.1,0.1,0.1);
	
	glEnableClientState(GL_TEXTURE_COORD_ARRAY);	
	glTexCoordPointer(2, GL_FLOAT, sizeof(GfxVertex), &m_vertices[0].m_uv[0]);
	
    glEnableClientState(GL_VERTEX_ARRAY);
    
    glColor4f(1, 1, 1, 1);
    glVertexPointer(3, GL_FLOAT, sizeof(GfxVertex), &m_vertices[0].m_position.getX());
 //   glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	
	for (int i=0;i<m_indices.size();i++)
	{
		if (m_indices[i] > m_vertices.size())
		{
			printf("out of bounds: m_indices[%d]=%d but m_vertices.size()=%d",i,m_indices[i],m_vertices.size());
		}
	}
	glDrawElements(GL_TRIANGLES,m_indices.size(),GL_UNSIGNED_SHORT,&m_indices[0]);

	glPopMatrix();
#else
	  // Load the vertex position
   glVertexAttribPointer ( positionLoc, 3, GL_FLOAT, GL_FALSE, sizeof(GfxVertex), &m_vertices[0].m_position.getX() );
   glVertexAttribPointer ( normalLoc, 3, GL_FLOAT, GL_FALSE, sizeof(GfxVertex), &m_vertices[0].m_normal.getX() );

   // Load the texture coordinate
   glVertexAttribPointer ( texCoordLoc, 2, GL_FLOAT,GL_FALSE, sizeof(GfxVertex), &m_vertices[0].m_uv[0] );

//		glDisable(GL_TEXTURE_2D);

	   glEnableVertexAttribArray ( positionLoc );
		glEnableVertexAttribArray ( normalLoc );

		glEnableVertexAttribArray ( texCoordLoc );
   // Bind the texture
  
//   glBindTexture ( GL_TEXTURE_2D, textureId );

   // Set the sampler texture unit to 0
   glUniform1i ( samplerLoc, 0 );


 
	float mat1[16];
	if (m_colObj)
		m_colObj->getWorldTransform().getOpenGLMatrix(mat1);
	else
		m_worldTransform.getOpenGLMatrix(mat1);
		
	

	glUniformMatrix4fv(modelMatrix,1,GL_FALSE,mat1);

	glDrawElements(GL_TRIANGLES,m_indices.size(),GL_UNSIGNED_SHORT,&m_indices[0]);
//	glLineWidth(3.0);
//	glDrawElements(GL_LINES,m_indices.size(),GL_UNSIGNED_SHORT,&m_indices[0]);

	
#endif

	checkGlError("1");
}

btAlignedObjectArray<GfxObject> s_graphicsObjects;



float projMat[16];

btDiscreteDynamicsWorld* dynWorld = 0;
btDefaultCollisionConfiguration* collisionConfiguration = 0;
btCollisionDispatcher* dispatcher = 0;
btDbvtBroadphase* broadphase = 0;
btSequentialImpulseConstraintSolver* solver = 0;



///
// Initialize the shader and program object
//
bool setupGraphics(int screenWidth, int screenHeight) 
{

		///collision configuration contains default setup for memory, collision setup
		collisionConfiguration = new btDefaultCollisionConfiguration();

		///use the default collision dispatcher. For parallel processing you can use a diffent dispatcher (see Extras/BulletMultiThreaded)
		dispatcher = new	btCollisionDispatcher(collisionConfiguration);

		broadphase = new btDbvtBroadphase();

		///the default constraint solver. For parallel processing you can use a different solver (see Extras/BulletMultiThreaded)
		solver = new btSequentialImpulseConstraintSolver;
	
		dynWorld = new btDiscreteDynamicsWorld(dispatcher,broadphase,solver,collisionConfiguration);
		dynWorld ->setGravity(btVector3(0,-10,0));

	 glViewport ( 0, 0, screenWidth,screenHeight );

	 checkGlError("glView");
	glEnable(GL_DEPTH_TEST);
	 checkGlError("glView");

    printGLString("Version", GL_VERSION);
    printGLString("Vendor", GL_VENDOR);
    printGLString("Renderer", GL_RENDERER);
    printGLString("Extensions", GL_EXTENSIONS);

   GLbyte vShaderStr[] =  
	  "uniform mat4 modelMatrix;\n"
	  "uniform mat4 viewMatrix;\n"
	  "uniform mat4 projectionMatrix;\n"
	  "attribute vec4 a_position;   \n"
	  "attribute vec4 a_normal;   \n"
	  "attribute vec2 a_texCoord;   \n"
      "varying vec2 v_texCoord;     \n"
      "varying vec4 v_normal;     \n"
      "void main()                  \n"
      "{                            \n"
	  "   gl_Position = (projectionMatrix*viewMatrix*modelMatrix)*a_position; \n"
      "   v_texCoord = a_texCoord;  \n"
	 "vec4 mnorm = vec4(a_normal.xyz,0.0);\n"
	 "mat3 normalMatrix = mat3( modelMatrix);\n"
	  "   v_normal = vec4(normalMatrix * a_normal.xyz,0.);\n"//\\(viewMatrix*modelMatrix)*mnorm;  \n"
      "}                            \n";
   
   GLbyte fShaderStr[] =  
      "precision mediump float;                            \n"
      "varying vec2 v_texCoord;                            \n"
		"varying vec4 v_normal;                            \n"
      "uniform sampler2D s_texture;                        \n"
      "void main()                                         \n"
      "{                                                   \n"
	  " vec4 lightDir = vec4(.0,.5,1.0,1.0);\n"
	  "float intensity=1.0;\n"
		"intensity = max(dot(lightDir,normalize(v_normal)),0.0);\n"
    //  "  gl_FragColor = vec4(0.0,intensity*1.0,intensity*1.0,intensity*1.0);\n"
//    "  gl_FragColor = texture2D( s_texture, v_texCoord );\n"
		"  gl_FragColor = vec4(0.0,intensity*1.0,intensity*1.0,1.0);\n"
"}                                                   \n";


// for wireframe, use white color
//	  "  gl_FragColor = vec4(1.0,1.0,1.0,1.0);\n"

   // Load the shaders and get a linked program object
//   programObject = esLoadProgram ((const char*)vShaderStr, (const char*)fShaderStr );
     programObject =
      shader_util::CreateProgramFromVertexAndFragmentShaders(
      (const char*)vShaderStr, (const char*)fShaderStr);

	 checkGlError("program");

   // Get the attribute locations
   positionLoc = glGetAttribLocation ( programObject, "a_position" );
   normalLoc = glGetAttribLocation ( programObject, "a_normal" );
   texCoordLoc = glGetAttribLocation ( programObject, "a_texCoord" );


	 checkGlError("texCoordLoc ");
   
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

//   createWorld();

   
	static const GLfloat cube_normals[] = {
    // Vertex coordinates interleaved with color values
    // Bottom//-1
    0.f, -1.f, 0.f,
	0.f, -1.f, 0.f,
	0.f, -1.f, 0.f,
	0.f, -1.f, 0.f,
	// Top
	0.f, 1.f, 0.f,
	0.f, 1.f, 0.f,
	0.f, 1.f, 0.f,
	0.f, 1.f, 0.f,
    // Back//-1
	0,0,-1,
0,0,-1,
0,0,-1,
0,0,-1,
    // Front
	0,0,1,
	0,0,1,
	0,0,1,
	0,0,1,

	// Left
	-1.f,0,0,
	-1.f,0,0,
	-1.f,0,0,
	-1.f,0,0,

    // Right
    1,0,0,
	1,0,0,
	1,0,0,
	1,0,0	
	};

	static const GLfloat cube_vertices[] = {
    // Vertex coordinates interleaved with color values
    // Bottom
    -0.5f, -0.5f, -0.5f,
    -0.5f, -0.5f, 0.5f,
    0.5f, -0.5f, 0.5f,
    0.5f, -0.5f, -0.5f,
    // Top
    -0.5f, 0.5f, -0.5f,
    -0.5f, 0.5f, 0.5f,
    0.5f, 0.5f, 0.5f,
    0.5f, 0.5f, -0.5f,
    // Back
    -0.5f, -0.5f, -0.5f,
    -0.5f, 0.5f, -0.5f,
    0.5f, 0.5f, -0.5f,
    0.5f, -0.5f, -0.5f,
    // Front
    -0.5f, -0.5f, 0.5f,
    -0.5f, 0.5f, 0.5f,
    0.5f, 0.5f, 0.5f,
    0.5f, -0.5f, 0.5f,
    // Left
    -0.5f, -0.5f, -0.5f,
    -0.5f, -0.5f, 0.5f,
    -0.5f, 0.5f, 0.5f,
    -0.5f, 0.5f, -0.5f,
    // Right
    0.5f, -0.5f, -0.5f,
    0.5f, -0.5f, 0.5f,
    0.5f, 0.5f, 0.5f,
    0.5f, 0.5f, -0.5f
  };



  static const GLushort cube_indices[] = {
    // Bottom
    0, 2, 1,
    0, 3, 2,
    // Top
    4, 5, 6,
    4, 6, 7,
    // Back
    8, 9, 10,
    8, 10, 11,
    // Front
    12, 15, 14,
    12, 14, 13,
    // Left
    16, 17, 18,
    16, 18, 19,
    // Right
    20, 23, 22,
    20, 22, 21
  };



		GfxObject* mesh = new GfxObject(0,0);
		mesh->m_vertices.resize(24);
		mesh->m_indices.resize(36);
		
		for (int i=0;i<24;i++)
		{
			mesh->m_vertices[i].m_position.setValue(cube_vertices[i*3],cube_vertices[i*3+1],cube_vertices[i*3+2]);
			mesh->m_vertices[i].m_normal.setValue(cube_normals[i*3],cube_normals[i*3+1],cube_normals[i*3+2]);
			mesh->m_vertices[i].m_normal.normalize();
		}

		for (int j=0;j<6;j++)
		{
				mesh->m_vertices[0+j*4].m_uv[0] = 0.f;
				mesh->m_vertices[0+j*4].m_uv[1] = 0.f;
				
				mesh->m_vertices[0+j*4+1].m_uv[0] = 0.f;
				mesh->m_vertices[0+j*4+1].m_uv[1] = 1.f;
				
				mesh->m_vertices[0+j*4+2].m_uv[0] = 1.f;
				mesh->m_vertices[0+j*4+2].m_uv[1] = 1.f;
				

				mesh->m_vertices[0+j*4+3].m_uv[0] = 1.f;
				mesh->m_vertices[0+j*4+3].m_uv[1] = 0.f;
				
		}			

		for (int i=0;i<36;i++)
			mesh->m_indices[i] = cube_indices[i];
			
		mesh->m_worldTransform.setIdentity();
		mesh->m_worldTransform.setOrigin(btVector3(0,1,0));

		GLubyte*	image=new GLubyte[256*256*3];
		for(int y=0;y<256;++y)
		{
			const int	t=y>>5;
			GLubyte*	pi=image+y*256*3;
			for(int x=0;x<256;++x)
			{
				const int		s=x>>5;
				const GLubyte	b=180;					
				GLubyte			c=b+((s+t&1)&1)*(255-b);
				pi[0]=0;
				pi[1]=255;
				pi[2]=255;
				pi+=3;
			}
		}
	btBoxShape* colShape = new btBoxShape(btVector3(0.5,0.5,0.5));
//	colShape->initializePolyhedralFeatures();

		mesh->m_texture = new BasicTexture(image,256,256);

		for (int i=0;i<100;i++)
		{		
			mesh->m_worldTransform.setIdentity();
			mesh->m_worldTransform.setOrigin(btVector3(0,i,0));
		btScalar mass = 1.f;
				btVector3 inertia;
				colShape->calculateLocalInertia(mass,inertia);
				btRigidBody* body = new btRigidBody(mass,0,colShape,inertia);
				body->setWorldTransform(mesh->m_worldTransform);
				dynWorld->addRigidBody(body);
		mesh->m_colObj = body;
		s_graphicsObjects.push_back(*mesh);
	}
		
	
		checkGlError("bla21");

{
	btBoxShape* colShape = new btBoxShape(btVector3(500.0,1.,500.0));
//	colShape->initializePolyhedralFeatures();

	//	btCollisionShape* colShape = new btStaticPlaneShape(btVector3(0,1,0),0);
	
	btVector3 inertia(0,0,0);
	btRigidBody* body = new btRigidBody(0.f,0,colShape,inertia);

	btTransform tr;
	tr.setIdentity();
	tr.setOrigin(btVector3(0,-1,0));

	body->setWorldTransform(tr);
	dynWorld->addRigidBody(body);
}
	
	
	mesh->m_worldTransform.setOrigin(btVector3(1,1,1));
		


   return true;
}

///
// Draw a triangle using the shader pair created in Init()
//
void renderFrame() 
{
	 
	 dynWorld->stepSimulation(1.0/60.f);
	 
  glClearColor ( 0.6f, 0.6f, 0.6f, 0.f );
	checkGlError("glClearColor");


      
  
   // Clear the color buffer
   glClear ( GL_COLOR_BUFFER_BIT |GL_DEPTH_BUFFER_BIT);

   // Use the program object
   glUseProgram ( programObject );
	checkGlError("glUseProgram ");

   // Load the vertex position
//   glVertexAttribPointer ( positionLoc, 3, GL_FLOAT, 
  //                         GL_FALSE, 5 * sizeof(GLfloat), vVertices );
   // Load the texture coordinate
   //glVertexAttribPointer ( texCoordLoc, 2, GL_FLOAT,
     //                      GL_FALSE, 5 * sizeof(GLfloat), &vVertices[3] );

   glEnableVertexAttribArray ( positionLoc );
//	glEnableVertexAttribArray ( normalLoc );
   glEnableVertexAttribArray ( texCoordLoc );
	checkGlError("glEnableVertexAttribArray");

//   glDisable(GL_TEXTURE_2D);
//   // Bind the texture
   glActiveTexture ( GL_TEXTURE0 );
   glBindTexture ( GL_TEXTURE_2D, textureId );

   // Set the sampler texture unit to 0
   glUniform1i ( samplerLoc, 0 );

	checkGlError("glUniform1i");
 
   btVector3 m_cameraPosition(1,12,15);
   btVector3 m_cameraTargetPosition(0,5,0);
   btVector3 m_cameraUp(0,1,0);//1,0);//1,0);
    float mat2[16];

//#define USE_CAM_FROM_FILE 1
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
	checkGlError("glUniformMatrix4fv");


	for (int i=0;i<s_graphicsObjects.size();i++)
	{
		s_graphicsObjects[i].render(positionLoc,normalLoc, texCoordLoc,samplerLoc,modelMatrix);
	}

	checkGlError("render");
	glFlush();
	
}


void shutdownGraphics()
{

}






namespace tumbler {

static const size_t kVertexCount = 24;
static const int kIndexCount = 36;

#ifdef __native_client__
Cube::Cube(SharedOpenGLContext opengl_context)
    : opengl_context_(opengl_context),
#else
Cube::Cube():
#endif
      width_(1),
      height_(1) {
  eye_[0] = eye_[1] = 0.0f;
  eye_[2] = 2.0f;
  orientation_[0] = 0.0f;
  orientation_[1] = 0.0f;
  orientation_[2] = 0.0f;
  orientation_[3] = 1.0f;
}

Cube::~Cube() {
#ifdef ORIGINAL
  glDeleteBuffers(3, cube_vbos_);
  glDeleteProgram(shader_program_object_);
#endif

}

void Cube::PrepareOpenGL() {
#ifdef ORIGINAL
  CreateShaders();
  CreateCube();
  glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
  glEnable(GL_DEPTH_TEST);
#else
	setupGraphics(width_, height_);
#endif
  
}

void Cube::Resize(int width, int height) 
{
  width_ = std::max(width, 1);
  height_ = std::max(height, 1);
  // Set the viewport
  glViewport(0, 0, width_, height_);
  // Compute the perspective projection matrix with a 60 degree FOV.
  GLfloat aspect = static_cast<GLfloat>(width_) / static_cast<GLfloat>(height_);
  transform_4x4::LoadIdentity(perspective_proj_);
  transform_4x4::Perspective(perspective_proj_, 60.0f, aspect, 1.0f, 20.0f);
}

void Cube::Draw() {

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

#ifdef ORIGINAL
  // Compute a new model-view matrix, then use that to make the composite
  // model-view-projection matrix: MVP = MV . P.
  GLfloat model_view[16];
  ComputeModelViewTransform(model_view);
  transform_4x4::Multiply(mvp_matrix_, model_view, perspective_proj_);

  glBindBuffer(GL_ARRAY_BUFFER, cube_vbos_[0]);
  glUseProgram(shader_program_object_);
  glEnableVertexAttribArray(position_location_);
  glVertexAttribPointer(position_location_,
                        3,
                        GL_FLOAT,
                        GL_FALSE,
                        3 * sizeof(GLfloat),
                        NULL);
  glEnableVertexAttribArray(color_location_);
  glBindBuffer(GL_ARRAY_BUFFER, cube_vbos_[1]);
  glVertexAttribPointer(color_location_,
                        3,
                        GL_FLOAT,
                        GL_FALSE,
                        3 * sizeof(GLfloat),
                        NULL);
  glUniformMatrix4fv(mvp_location_, 1, GL_FALSE, mvp_matrix_);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, cube_vbos_[2]);
  glDrawElements(GL_TRIANGLES, kIndexCount, GL_UNSIGNED_SHORT, 0);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
#else
	renderFrame();  
#endif
  
}

bool Cube::CreateShaders() {
  return true;
}

void Cube::CreateCube() {
  static const GLfloat cube_vertices[] = {
    // Vertex coordinates interleaved with color values
    // Bottom
    -0.5f, -0.5f, -0.5f,
    -0.5f, -0.5f, 0.5f,
    0.5f, -0.5f, 0.5f,
    0.5f, -0.5f, -0.5f,
    // Top
    -0.5f, 0.5f, -0.5f,
    -0.5f, 0.5f, 0.5f,
    0.5f, 0.5f, 0.5f,
    0.5f, 0.5f, -0.5f,
    // Back
    -0.5f, -0.5f, -0.5f,
    -0.5f, 0.5f, -0.5f,
    0.5f, 0.5f, -0.5f,
    0.5f, -0.5f, -0.5f,
    // Front
    -0.5f, -0.5f, 0.5f,
    -0.5f, 0.5f, 0.5f,
    0.5f, 0.5f, 0.5f,
    0.5f, -0.5f, 0.5f,
    // Left
    -0.5f, -0.5f, -0.5f,
    -0.5f, -0.5f, 0.5f,
    -0.5f, 0.5f, 0.5f,
    -0.5f, 0.5f, -0.5f,
    // Right
    0.5f, -0.5f, -0.5f,
    0.5f, -0.5f, 0.5f,
    0.5f, 0.5f, 0.5f,
    0.5f, 0.5f, -0.5f
  };

  static const GLfloat cube_colors[] = {
    // Vertex coordinates interleaved with color values
    // Bottom
    1.0, 0.0, 0.0,
    1.0, 0.0, 0.0,
    1.0, 0.0, 0.0,
    1.0, 0.0, 0.0,
    // Top
    0.0, 1.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 1.0, 0.0,
    // Back
    0.0, 0.0, 1.0,
    0.0, 0.0, 1.0,
    0.0, 0.0, 1.0,
    0.0, 0.0, 1.0,
    // Front
    1.0, 0.0, 1.0,
    1.0, 0.0, 1.0,
    1.0, 0.0, 1.0,
    1.0, 0.0, 1.0,
    // Left
    1.0, 1.0, 0.0,
    1.0, 1.0, 0.0,
    1.0, 1.0, 0.0,
    1.0, 1.0, 0.0,
    // Right
    0.0, 1.0, 1.0,
    0.0, 1.0, 1.0,
    0.0, 1.0, 1.0,
    0.0, 1.0, 1.0
  };

  static const GLushort cube_indices[] = {
    // Bottom
    0, 2, 1,
    0, 3, 2,
    // Top
    4, 5, 6,
    4, 6, 7,
    // Back
    8, 9, 10,
    8, 10, 11,
    // Front
    12, 15, 14,
    12, 14, 13,
    // Left
    16, 17, 18,
    16, 18, 19,
    // Right
    20, 23, 22,
    20, 22, 21
  };

  // Generate the VBOs and upload them to the graphics context.
  glGenBuffers(3, cube_vbos_);
  glBindBuffer(GL_ARRAY_BUFFER, cube_vbos_[0]);
  glBufferData(GL_ARRAY_BUFFER,
               kVertexCount * sizeof(GLfloat) * 3,
               cube_vertices,
               GL_STATIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, cube_vbos_[1]);
  glBufferData(GL_ARRAY_BUFFER,
               kVertexCount * sizeof(GLfloat) * 3,
               cube_colors,
               GL_STATIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, cube_vbos_[2]);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER,
               kIndexCount * sizeof(GL_UNSIGNED_SHORT),
               cube_indices,
               GL_STATIC_DRAW);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void Cube::ComputeModelViewTransform(GLfloat* model_view) {
  // This method takes into account the possiblity that |orientation_|
  // might not be normalized.
  double sqrx = orientation_[0] * orientation_[0];
  double sqry = orientation_[1] * orientation_[1];
  double sqrz = orientation_[2] * orientation_[2];
  double sqrw = orientation_[3] * orientation_[3];
  double sqrLength = 1.0 / (sqrx + sqry + sqrz + sqrw);

  transform_4x4::LoadIdentity(model_view);
  model_view[0] = (sqrx - sqry - sqrz + sqrw) * sqrLength;
  model_view[5] = (-sqrx + sqry - sqrz + sqrw) * sqrLength;
  model_view[10] = (-sqrx - sqry + sqrz + sqrw) * sqrLength;

  double temp1 = orientation_[0] * orientation_[1];
  double temp2 = orientation_[2] * orientation_[3];
  model_view[1] = 2.0 * (temp1 + temp2) * sqrLength;
  model_view[4] = 2.0 * (temp1 - temp2) * sqrLength;

  temp1 = orientation_[0] * orientation_[2];
  temp2 = orientation_[1] * orientation_[3];
  model_view[2] = 2.0 * (temp1 - temp2) * sqrLength;
  model_view[8] = 2.0 * (temp1 + temp2) * sqrLength;
  temp1 = orientation_[1] * orientation_[2];
  temp2 = orientation_[0] * orientation_[3];
  model_view[6] = 2.0 * (temp1 + temp2) * sqrLength;
  model_view[9] = 2.0 * (temp1 - temp2) * sqrLength;
  model_view[3] = 0.0;
  model_view[7] = 0.0;
  model_view[11] = 0.0;

  // Concatenate the translation to the eye point.
  model_view[12] = -eye_[0];
  model_view[13] = -eye_[1];
  model_view[14] = -eye_[2];
  model_view[15] = 1.0;
}

}  // namespace tumbler


