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
#ifndef _SHADER_H_
#define _SHADER_H_

//#include "Context.h"
//#include "GraphicsDevice.h"
//#include <OpenGLES/EAGL.h>
#include <TargetConditionals.h>
#include <Availability.h>
#if __IPHONE_OS_VERSION_MAX_ALLOWED >= 30000
#include <OpenGLES/ES2/gl.h>
#include <OpenGLES/ES2/glext.h>


#include <string>
//#include "../Error.h"

/**************************ShaderLoadSourceFromMemory*************************
 pszShaderCode		shader source code
 Type			GL_VERTEX_SHADER or GL_FRAGMENT_SHADER
 pObject		the resulting shader object
 pReturnError		the error message if it failed
 
 Returns SUCCESS on success and FAIL on failure (also fills the str string),
 currently the str string is not used.
*****************************************************************************/
unsigned int ShaderLoadSourceFromMemory(	const char* pszShaderCode, 
											const GLenum Type, 
											GLuint* const pObject);
											//string const pReturnError);

/**************************ShaderLoadBinaryFromMemory*************************
 ShaderData		shader compiled binary data
 Size			shader binary data size (bytes)
 Type			GL_VERTEX_SHADER or GL_FRAGMENT_SHADER
 Format			shader binary format
 pObject		the resulting shader object
 
 Returns SUCCESS on success and FAIL on failure (also fills the str string),
 currently the str string is not used.
*****************************************************************************/
unsigned int ShaderLoadBinaryFromMemory(	const void*  const ShaderData, 
						const size_t Size, 
						const GLenum Type, 
						const GLenum Format, 
						GLuint*  const pObject); 

/*!***************************************************************************
 ShaderLoadFromFile
 pszBinFile		binary shader filename
 pszSrcFile		source shader filename
 Type			type of shader (GL_VERTEX_SHADER or GL_FRAGMENT_SHADER)
 Format			shader binary format, or 0 for source shader
 pObject		the resulting shader object
 pReturnError		the error message if it failed
 pContext		Context
 
 Returns SUCCESS on success and FAIL on failure (also fills pReturnError),
 and then Loads a shader file into memory and passes it to the GL.
*****************************************************************************/
unsigned int ShaderLoadFromFile(	const char* const pszBinFile, 
					const char* const pszSrcFile, 
					const GLenum Type,
					const GLenum Format, 
					GLuint* const pObject);

/*!***************************************************************************
 @Function		CreateProgram
 @Output		pProgramObject			the created program object
 @Input			VertexShader			the vertex shader to link
 @Input			FragmentShader			the fragment shader to link
 @Input			pszAttribs				an array of attribute names
 @Input			i32NumAttribs			the number of attributes to bind
 @Output		pReturnError			the error message if it failed
 @Returns		PVR_SUCCESS on success, PVR_FAIL if failure
 @Description	Links a shader program.
*****************************************************************************/
unsigned int CreateProgram(	GLuint* const pProgramObject, 
						    const GLuint VertexShader, 
						    const GLuint FragmentShader, 
						    const char** const pszAttribs,
						    const int i32NumAttribs);

/*!***************************************************************************
 @Function		TestGLError
 @Input			pszLocation				an array of attribute names
 @Output		pszLocation				the error message if it failed
 @Returns		true on success, false on failure
 @Description	Checks to make sure no processes irrecoverably fail.
 *****************************************************************************/
bool TestGLError(const char* pszLocation);

#endif

/*****************************************************************************
 End of file (Shader.h)
*****************************************************************************/
#endif
