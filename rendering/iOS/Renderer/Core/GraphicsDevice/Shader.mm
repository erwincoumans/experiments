/******************************************************************************

 @File         Shader.cpp

 @Title        PFX file parser.

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     ANSI compatible

 @Description  Shader handling for OpenGL ES 2.0

******************************************************************************/

#include <string>
#include "Shader.h"
#include "ResourceFile.h"

#define GL_SUCCESS		0x3000
#define SUCCESS			1
#define FAIL			0

#if __IPHONE_OS_VERSION_MAX_ALLOWED >= 30000
/*!***************************************************************************
 @Function		ShaderLoadSourceFromMemory
 @Input			pszShaderCode	shader source code
 @Input			Type			type of shader (GL_VERTEX_SHADER or GL_FRAGMENT_SHADER)
 @Output		pObject			the resulting shader object
 @Output		pReturnError	the error message if it failed
 @Return		SUCCESS on success and FAIL on failure (also fills the str string)
 @Description	Loads a shader source code into memory and compiles it.
*****************************************************************************/
unsigned int ShaderLoadSourceFromMemory(	const char* pszShaderCode, 
											const GLenum Type, 
											GLuint* const pObject)
											//string const pReturnError)
{
	// Create the shader object.
    *pObject = glCreateShader(Type);
	
	// Load the source code into it.
    glShaderSource(*pObject, 1, &pszShaderCode, NULL);
	
	// Compile the source code.
    glCompileShader(*pObject);

	// Test if compilation succeeded.
	GLint bShaderCompiled;
    glGetShaderiv(*pObject, GL_COMPILE_STATUS, &bShaderCompiled);
	if (!bShaderCompiled)
	{
		// There was an error here, first get the length of the log message.
		int i32InfoLogLength, i32CharsWritten;
		glGetShaderiv(*pObject, GL_INFO_LOG_LENGTH, &i32InfoLogLength);
		
		// Allocate enough space for the message, and retrieve it.
		char* pszInfoLog = new char[i32InfoLogLength];
        glGetShaderInfoLog(*pObject, i32InfoLogLength, &i32CharsWritten, pszInfoLog);
		
		// Displays the error!
		//printf("Failed to compile fragment shader: %s\n", pszInfoLog);
		delete [] pszInfoLog;
		
		// Delete shader.
		glDeleteShader(*pObject);
		
		// Return false, couldn't compile.
		return FAIL;
	}
	
	return SUCCESS;
}

/*!***************************************************************************
 @Function		ShaderLoadBinaryFromMemory
 @Input			ShaderData		shader compiled binary data
 @Input			Size			size of shader binary data in bytes
 @Input			Type			type of shader (GL_VERTEX_SHADER or GL_FRAGMENT_SHADER)
 @Input			Format			shader binary format
 @Output		pObject			the resulting shader object
 @Output		pReturnError	the error message if it failed
 @Return		SUCCESS on success and FAIL on failure (also fills the str string)
 @Description	Takes a shader binary from memory and passes it to the GL.
*****************************************************************************/
unsigned int ShaderLoadBinaryFromMemory(	const void* const ShaderData, 
											const size_t Size, 
											const GLenum Type, 
											const GLenum Format, 
											GLuint* const pObject) 
											//char* const pReturnError)
{
	// Create and compile the shader object
    *pObject = glCreateShader(Type);
    glShaderBinary(1, pObject, Format, ShaderData, (GLint)Size);
    if (glGetError() != GL_NO_ERROR)
    {
    	//*pReturnError = CString("Failed to load binary shader\n");
    	glDeleteShader(*pObject);
    	return FAIL;
    }
    return SUCCESS;
}

/*!***************************************************************************
 @Function		ShaderLoadFromFile
 @Input			pszBinFile		binary shader filename
 @Input			pszSrcFile		source shader filename
 @Input			Type			type of shader (GL_VERTEX_SHADER or GL_FRAGMENT_SHADER)
 @Input			Format			shader binary format, or 0 for source shader
 @Output		pObject			the resulting shader object
 @Output		pReturnError	the error message if it failed
 @Input			pContext		Context
 @Return		SUCCESS on success and FAIL on failure (also fills pReturnError)
 @Description	Loads a shader file into memory and passes it to the GL.
*****************************************************************************/
unsigned int ShaderLoadFromFile(	const char* const pszBinFile, 
									const char* const pszSrcFile, 
									const GLenum Type,
									const GLenum Format, 
									GLuint* const pObject)
									//CString* const pReturnError, 
									//const SContext* const pContext)
{
    //*pReturnError = "";

    if(Format)
    {
        CPVRTResourceFile ShaderFile(pszBinFile);
        if (ShaderFile.IsOpen())
        {
                //if(ShaderLoadBinaryFromMemory(ShaderFile.DataPtr(), ShaderFile.Size(), Type, Format, pObject, pReturnError) == SUCCESS)
				if(ShaderLoadBinaryFromMemory(ShaderFile.DataPtr(), ShaderFile.Size(), Type, Format, pObject) == SUCCESS)
                        return SUCCESS;
        }

        //*pReturnError += CString("Failed to open shader ") + pszBinFile + "\n";
    }


    CPVRTResourceFile ShaderFile(pszSrcFile);
    if (!ShaderFile.IsOpen())
    {
            //*pReturnError += CString("Failed to open shader ") + pszSrcFile + "\n";
            return FAIL;
    }

    //return ShaderLoadSourceFromMemory(ShaderFile.StringPtr(), Type, pObject, pReturnError);
	return ShaderLoadSourceFromMemory(ShaderFile.StringPtr(), Type, pObject);
}


/*!***************************************************************************
 @Function		CreateProgram
 @Output		pProgramObject			the created program object
 @Input			VertexShader			the vertex shader to link
 @Input			FragmentShader			the fragment shader to link
 @Input			pszAttribs				an array of attribute names
 @Input			i32NumAttribs			the number of attributes to bind
 @Output		pReturnError			the error message if it failed
 @Returns		SUCCESS on success, FAIL if failure
 @Description	Links a shader program.
*****************************************************************************/
unsigned int CreateProgram(	GLuint* const pProgramObject, 
							const GLuint VertexShader, 
							const GLuint FragmentShader, 
							const char** const pszAttribs,
							const int i32NumAttribs)
							//CString* const pReturnError)
{
	// Create the shader program.
	*pProgramObject = glCreateProgram();

	// Attach the fragment and vertex shaders to it.
	glAttachShader(*pProgramObject, FragmentShader);
	glAttachShader(*pProgramObject, VertexShader);

	// For every member in pszAttribs, bind the proper attributes.
	for (int i = 0; i < i32NumAttribs; ++i)
	{
		glBindAttribLocation(*pProgramObject, i, pszAttribs[i]);
	}

	// Link the program object
	glLinkProgram(*pProgramObject);
	
	// Check if linking succeeded.
	GLint bLinked;
	glGetProgramiv(*pProgramObject, GL_LINK_STATUS, &bLinked);
	if (!bLinked)
	{
		int i32InfoLogLength, i32CharsWritten;
		glGetProgramiv(*pProgramObject, GL_INFO_LOG_LENGTH, &i32InfoLogLength);
		char* pszInfoLog = new char[i32InfoLogLength];
		glGetProgramInfoLog(*pProgramObject, i32InfoLogLength, &i32CharsWritten, pszInfoLog);
		//*pReturnError = CString("Failed to link: ") + pszInfoLog + "\n";
		delete [] pszInfoLog;
		return FAIL;
	}

	// Actually use the created program.
	glUseProgram(*pProgramObject);

	return SUCCESS;
}

/*!***************************************************************************
 @Function		TestGLError
 @Output		pszLocation			the error message if it failed
 @Returns		SUCCESS on success, FAIL if failure
 @Description	Tests for last state encountered.
 *****************************************************************************/
bool TestGLError(const char* pszLocation)
{
	/*
	 TestGLError returns the last error that has happened using egl,
	 not the status of the last called function. The user has to
	 check after every single egl call or at least once every frame.
	 */
	GLint iErr = glGetError();
	if (iErr != GL_SUCCESS)
	{
		//printf("%s failed (%d).\n", pszLocation, iErr);
		return FAIL;
	}
	
	return SUCCESS;
}

/*****************************************************************************
 End of file (Shader.cpp)
*****************************************************************************/
#endif
