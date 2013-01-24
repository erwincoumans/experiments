/*
Oolong Engine for the iPhone / iPod touch
Copyright (c) 2007-2008 Wolfgang Engel  http://code.google.com/p/oolongengine/

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/
#ifndef GLESEXT_H_
#define GLESEXT_H_

#import <OpenGLES/EAGL.h>
#import <OpenGLES/ES1/gl.h>
#import <OpenGLES/ES1/glext.h>

#ifndef APIENTRY
#define APIENTRY
#endif
#ifndef GL_APIENTRY
#define GL_APIENTRY
#endif

#define GL_PVRTGLESEXT_VERSION 1

class OpenGLESExt
{
public:
	/*
		GL_OES_matrix_palette
	*/
	typedef void (GL_APIENTRY * PFNGLCURRENTPALETTEMATRIXOES) (GLuint matrixpaletteindex);
	typedef void (GL_APIENTRY * PFNGLLOADPALETTEFROMMODELVIEWMATRIXOES) (void);
	typedef void (GL_APIENTRY * PFNGLMATRIXINDEXPOINTEROES) (GLint size, GLenum type, GLsizei stride, const GLvoid *pointer);
	typedef void (GL_APIENTRY * PFNGLWEIGHTPOINTEROES) (GLint size, GLenum type, GLsizei stride, const GLvoid *pointer);

	/* Function pointers */
	PFNGLCURRENTPALETTEMATRIXOES			glCurrentPaletteMatrixOES;
	PFNGLLOADPALETTEFROMMODELVIEWMATRIXOES	glLoadPaletteFromModelViewMatrixOES;
	PFNGLMATRIXINDEXPOINTEROES				glMatrixIndexPointerOES;
	PFNGLWEIGHTPOINTEROES					glWeightPointerOES;

	/*
		GL_IMG_texture_env_enhanced_fixed_function
	*/
	/* Tokens */
	#define GL_MODULATE_COLOR_IMG						0x8C04
	#define GL_RECIP_ADD_SIGNED_ALPHA_IMG				0x8C05
	#define GL_TEXTURE_ALPHA_MODULATE_IMG				0x8C06
	#define GL_FACTOR_ALPHA_MODULATE_IMG				0x8C07
	#define GL_FRAGMENT_ALPHA_MODULATE_IMG				0x8C08
	#define GL_ADD_BLEND_IMG							0x8C09
	#ifndef GL_DOT3_RGBA
	#define GL_DOT3_RGBA								0x86AF
	#endif


	/*
		GL_IMG_texture_compression_pvrtc
	*/
	/* Tokens */
	#define GL_COMPRESSED_RGB_PVRTC_4BPPV1_IMG			0x8C00
	#define GL_COMPRESSED_RGB_PVRTC_2BPPV1_IMG			0x8C01
	#define GL_COMPRESSED_RGBA_PVRTC_4BPPV1_IMG			0x8C02
	#define GL_COMPRESSED_RGBA_PVRTC_2BPPV1_IMG			0x8C03


	/*
		GL_ARB_texture_env_combine
	*/
	/* Tokens */
	#define GL_COMBINE									0x8570
	#define GL_COMBINE_RGB								0x8571
	#define GL_COMBINE_ALPHA							0x8572
	#define GL_RGB_SCALE								0x8573
	#define GL_ADD_SIGNED								0x8574
	#define GL_INTERPOLATE								0x8575
	#define GL_CONSTANT									0x8576
	#define GL_PRIMARY_COLOR							0x8577
	#define GL_PREVIOUS									0x8578
	#define GL_SOURCE0_RGB								0x8580
	#define GL_SOURCE1_RGB								0x8581
	#define GL_SOURCE2_RGB								0x8582
	#define GL_SOURCE0_ALPHA							0x8588
	#define GL_SOURCE1_ALPHA							0x8589
	#define GL_SOURCE2_ALPHA							0x858A
	#define GL_OPERAND0_RGB								0x8590
	#define GL_OPERAND1_RGB								0x8591
	#define GL_OPERAND2_RGB								0x8592
	#define GL_OPERAND0_ALPHA							0x8598
	#define GL_OPERAND1_ALPHA							0x8599
	#define GL_OPERAND2_ALPHA							0x859A
	#define GL_SUBTRACT									0x84E7


	/*
		GL_ARB_texture_env_dot3
	*/
	/* Tokens */
	#define GL_DOT3_RGB									0x86AE
	#ifndef GL_DOT3_RGBA
	#define GL_DOT3_RGBA								0x86AF
	#endif


	/*
		GL_IMG_user_clip_plane
	*/
	/* Tokens */
	#define GL_MAX_CLIP_PLANES							0x0D32
	#define GL_CLIP_PLANE0								0x3000
	#define GL_CLIP_PLANE1								0x3001
	#define GL_CLIP_PLANE2								0x3002
	#define GL_CLIP_PLANE3								0x3003
	#define GL_CLIP_PLANE4								0x3004
	#define GL_CLIP_PLANE5								0x3005

	/* Typedefs */
	typedef void (APIENTRY * PFNGLCLIPPLANEFIMG) (GLenum p, GLfloat eqn[4]);
	typedef void (APIENTRY * PFNGLCLIPPLANEXIMG) (GLenum p, GLfixed eqn[4]);

	/* Function pointers */
	PFNGLCLIPPLANEFIMG	glClipPlanefIMG;
	PFNGLCLIPPLANEXIMG	glClipPlanexIMG;


	/*
		GL_IMG_vertex_program
	*/
	/* Tokens */
	#define GL_VERTEX_PROGRAM_ARB						0x8620
	#define GL_PROGRAM_STRING_ARB						0x8628
	#define GL_PROGRAM_ERROR_STRING_ARB					0x8874
	#define GL_MAX_PROGRAM_ATTRIBS_ARB					0x88AD
	#define GL_MAX_PROGRAM_PARAMETERS_ARB				0x88A9
	#define GL_MAX_PROGRAM_LOCAL_PARAMETERS_ARB			0x88B4
	#define GL_MAX_PROGRAM_ENV_PARAMETERS_ARB			0x88B5
	#define GL_MAX_VERTEX_ATTRIBS_ARB					0x8869
	#define GL_MAX_PROGRAM_MATRICES_ARB					0x862F
	#define GL_MAX_PROGRAM_MATRIX_STACK_DEPTH_ARB		0x862E
	#define GL_MATRIX0_ARB								0x88C0
	#define GL_MATRIX1_ARB								0x88C1
	#define GL_MATRIX2_ARB								0x88C2
	#define GL_MATRIX3_ARB								0x88C3
	#define GL_MATRIX4_ARB								0x88C4
	#define GL_MATRIX5_ARB								0x88C5
	#define GL_MATRIX6_ARB								0x88C6
	#define GL_MATRIX7_ARB								0x88C7
	#define GL_PROGRAM_FORMAT_BINARY_IMG				0x8C0B
	#define GL_UNSIGNED_BYTE_NORM_IMG					0x8C0C
	/* IMG_texture_stream */
	#define GL_TEXTURE_STREAM_IMG						0x8C0D
	#define GL_TEXTURE_NUM_STREAM_DEVICES_IMG			0x8C0E
	#define GL_TEXTURE_STREAM_DEVICE_WIDTH_IMG			0x8C0F
	#define GL_TEXTURE_STREAM_DEVICE_HEIGHT_IMG			0x8C10
	#define GL_TEXTURE_STREAM_DEVICE_FORMAT_IMG			0x8C11
	#define GL_TEXTURE_STREAM_DEVICE_NUM_BUFFERS_IMG	0x8C12

	/* Typedefs */
	typedef void (APIENTRY * PFNGLVERTEXATTRIBPOINTERARB) (GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, const GLvoid *pointer);
	typedef void (APIENTRY * PFNGLENABLEVERTEXATTRIBARRAYARB) (GLuint index);
	typedef void (APIENTRY * PFNGLDISABLEVERTEXATTRIBARRAYARB) (GLuint index);
	typedef void (APIENTRY * PFNGLPROGRAMSTRINGARB) (GLenum target, GLenum format, GLsizei len, const GLvoid *string);
	typedef void (APIENTRY * PFNGLBINDPROGRAMARB) (GLenum target, GLuint program);
	typedef void (APIENTRY * PFNGLDELETEPROGRAMSARB) (GLsizei n, const GLuint *programs);
	typedef void (APIENTRY * PFNGLGENPROGRAMSARB) (GLsizei n, GLuint *programs);
	typedef void (APIENTRY * PFNGLPROGRAMENVPARAMETER4FARB) (GLenum target, GLuint index, GLfloat x, GLfloat y, GLfloat z, GLfloat w);
	typedef void (APIENTRY * PFNGLPROGRAMENVPARAMETER4FVARB) (GLenum target, GLuint index, const GLfloat *params);
	typedef void (APIENTRY * PFNGLPROGRAMLOCALPARAMETER4FARB) (GLenum target, GLuint index, GLfloat x, GLfloat y, GLfloat z, GLfloat w);
	typedef void (APIENTRY * PFNGLPROGRAMLOCALPARAMETER4FVARB) (GLenum target, GLuint index, const GLfloat *params);
	typedef GLboolean (APIENTRY * PFNGLISPROGRAMARB) (GLuint program);
	typedef void (APIENTRY * PFNGLVERTEXATTRIB4FVARB) (GLuint index, const float *v);
	typedef void (APIENTRY * PFNGLVERTEXATTRIB4XVIMG) (GLuint index, const GLfixed *v);
	typedef void (APIENTRY * PFNGLPROGRAMLOCALPARAMETER4XIMG) (GLenum target, GLuint index, GLfixed x, GLfixed y, GLfixed z, GLfixed w);
	typedef void (APIENTRY * PFNGLPROGRAMLOCALPARAMETER4XVIMG) (GLenum target, GLuint index, const GLfixed *params);
	typedef void (APIENTRY * PFNGLPROGRAMENVPARAMETER4XIMG) (GLenum target, GLuint index, GLfixed x, GLfixed y, GLfixed z, GLfixed w);
	typedef void (APIENTRY * PFNGLPROGRAMENVPARAMETER4XVIMG) (GLenum target, GLuint index, const GLfixed *params);
	typedef void (APIENTRY * PFNGLDRAWTEXIOES) (GLint x, GLint y, GLint z, GLint width, GLint height);
	typedef void (APIENTRY * PFNGLDRAWTEXIVOES) (const GLint* coords);
	typedef void (APIENTRY * PFNGLDRAWTEXSOES) (GLshort x, GLshort y, GLshort z, GLshort width, GLshort height);
	typedef void (APIENTRY * PFNGLDRAWTEXSVOES) (const GLshort* coords);
	typedef void (APIENTRY * PFNGLDRAWTEXXOES) (GLfixed x, GLfixed y, GLfixed z, GLfixed width, GLfixed height);
	typedef void (APIENTRY * PFNGLDRAWTEXXVOES) (const GLfixed* coords);
	typedef void (APIENTRY * PFNGLDRAWTEXFOES) (GLfloat x, GLfloat y, GLfloat z, GLfloat width, GLfloat height);
	typedef void (APIENTRY * PFNGLDRAWTEXFVOES) (const GLfloat* coords);
	/* IMG_texture_stream */
	typedef void (APIENTRY * PFNGLGETTEXSTREAMDEVICEATTRIBIVIMG)(GLint device, GLenum pname, GLint *params);
	typedef void (APIENTRY * PFNGLTEXBINDSTREAMIMG)(GLint device, GLint deviceoffset);
	typedef const GLubyte * (APIENTRY * PFNGLGETTEXSTREAMDEVICENAMEIMG)(GLint device);
	/* GL_EXT_multi_draw_arrays */
	typedef void (APIENTRY * PFNGLMULTIDRAWELEMENTS) (GLenum mode, GLsizei *count, GLenum type, const GLvoid **indices, GLsizei primcount); // glvoid


	/* Function pointers */
	PFNGLVERTEXATTRIBPOINTERARB			glVertexAttribPointerARB;
	PFNGLENABLEVERTEXATTRIBARRAYARB		glEnableVertexAttribArrayARB;
	PFNGLDISABLEVERTEXATTRIBARRAYARB	glDisableVertexAttribArrayARB;
	PFNGLPROGRAMSTRINGARB				glProgramStringARB;
	PFNGLBINDPROGRAMARB					glBindProgramARB;
	PFNGLDELETEPROGRAMSARB				glDeleteProgramsARB;
	PFNGLGENPROGRAMSARB					glGenProgramsARB;
	PFNGLISPROGRAMARB					glIsProgramARB;
	PFNGLPROGRAMENVPARAMETER4FARB		glProgramEnvParameter4fARB;
	PFNGLPROGRAMENVPARAMETER4FVARB		glProgramEnvParameter4fvARB;
	PFNGLPROGRAMLOCALPARAMETER4FARB		glProgramLocalParameter4fARB;
	PFNGLPROGRAMLOCALPARAMETER4FVARB	glProgramLocalParameter4fvARB;
	PFNGLVERTEXATTRIB4FVARB				glVertexAttrib4fvARB;
	PFNGLVERTEXATTRIB4XVIMG				glVertexAttrib4xvIMG;
	PFNGLPROGRAMLOCALPARAMETER4XIMG		glProgramLocalParameter4xIMG;
	PFNGLPROGRAMLOCALPARAMETER4XVIMG	glProgramLocalParameter4xvIMG;
	PFNGLPROGRAMENVPARAMETER4XIMG		glProgramEnvParameter4xIMG;
	PFNGLPROGRAMENVPARAMETER4XVIMG		glProgramEnvParameter4xvIMG;
	PFNGLDRAWTEXIOES					glDrawTexiOES;
	PFNGLDRAWTEXIVOES					glDrawTexivOES;
	PFNGLDRAWTEXSOES					glDrawTexsOES;
	PFNGLDRAWTEXSVOES					glDrawTexsvOES;
	PFNGLDRAWTEXXOES					glDrawTexxOES;
	PFNGLDRAWTEXXVOES					glDrawTexxvOES;
	PFNGLDRAWTEXFOES					glDrawTexfOES;
	PFNGLDRAWTEXFVOES					glDrawTexfvOES;
	/* IMG_texture_stream */
	PFNGLGETTEXSTREAMDEVICEATTRIBIVIMG	glGetTexStreamDeviceAttribivIMG;
	PFNGLTEXBINDSTREAMIMG				glTexBindStreamIMG;
	PFNGLGETTEXSTREAMDEVICENAMEIMG		glGetTexStreamDeviceNameIMG;
	/* GL_EXT_multi_draw_arrays */
	PFNGLMULTIDRAWELEMENTS				glMultiDrawElementsEXT;

	// EGL_IMG_power_management
	#define EGL_CONTEXT_LOST_IMG						0x300E

public:
	void Init();

	// The recommended technique for querying OpenGL extensions;
	// from http://opengl.org/resources/features/OGLextensions/
	static bool IsGLExtensionSupported(const char *extension);
};


#endif 
