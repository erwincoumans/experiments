/******************************************************************************

 @File         PFXParserAPI.h

 @Title        PFX file parser.

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     Windows + Linux

 @Description  Declaration of PFX file parser

******************************************************************************/
#ifndef _PFXPARSERAPI_H_
#define _PFXPARSERAPI_H_

#if __IPHONE_OS_VERSION_MAX_ALLOWED >= 30000
#import <OpenGLES/ES2/gl.h>
#import <OpenGLES/ES2/glext.h>
#else
#import <OpenGLES/ES1/gl.h>
#import <OpenGLES/ES1/glext.h>
#endif


/****************************************************************************
** Structures
****************************************************************************/

/*! Application supplies an array of these so PFX can translate strings to numbers */
struct SPFXUniformSemantic
{
	const char		*p;	// String containing semantic
	unsigned int	n;	// Application-defined semantic value
};

/*! PFX returns an array of these to indicate GL locations & semantics to the application */
struct SPFXUniform
{
	unsigned int	nLocation;	// GL location of the Uniform
	unsigned int	nSemantic;	// Application-defined semantic value
	unsigned int	nIdx;		// Index; for example two semantics might be LIGHTPOSITION0 and LIGHTPOSITION1
};

/*! An array of these is gained from PFX so the application can fill in the texture handles*/
struct SPFXTexture
{
	const char	*p;		// texture FileName
	GLuint		ui;		// Loaded texture handle
};

/*!**************************************************************************
@Class CPFXEffect
@Brief PFX effect
****************************************************************************/
class CPFXEffect
{
public:
	//SContext	*m_psContext;
	CPFXParser	*m_pParser;
	unsigned int	m_nEffect;

	GLuint			m_uiProgram;		// Loaded program
	unsigned int	*m_pnTextureIdx;	// Array of indices into the texture array

	SPFXTexture	*m_psTextures;		// Array of loaded textures

public:
	/*!***************************************************************************
	@Function			CPFXEffect Blank Constructor
	@Description		Sets the context and initialises the member variables to zero.
	*****************************************************************************/
	CPFXEffect();

	/*!***************************************************************************
	@Function			CPFXEffect Constructor
	@Description		Sets the context and initialises the member variables to zero.
	*****************************************************************************/
	//CPFXEffect(SContext &sContext);

	/*!***************************************************************************
	@Function			CPFXEffect Destructor
	@Description		Calls Destroy().
	*****************************************************************************/
	~CPFXEffect();

	/*!***************************************************************************
	@Function			Load
	@Input				src					PFX Parser Object
	@Input				pszEffect			Effect name
	@Input				pszFileName			Effect file name
	@Output				pReturnError		Error string
	@Returns			EError			PVR_SUCCESS if load succeeded
	@Description		Loads the specified effect from the CPFXParser object.
						Compiles and links the shaders. Initialises texture data.
	*****************************************************************************/
	unsigned int Load(CPFXParser &src, const char * const pszEffect, const char * const pszFileName, string *pReturnError);

	/*!***************************************************************************
	@Function			Destroy
	@Description		Deletes the gl program object and texture data.
	*****************************************************************************/
	void Destroy();

	/*!***************************************************************************
	@Function			Activate
	@Returns			PVR_SUCCESS if activate succeeded
	@Description		Selects the gl program object and binds the textures.
	*****************************************************************************/
	unsigned int Activate();

	/*!***************************************************************************
	@Function			BuildUniformTable
	@Output				ppsUniforms					pointer to uniform data array
	@Output				pnUniformCount				pointer to number of uniforms
	@Output				pnUnknownUniformCount		pointer to number of unknown uniforms
	@Input				psUniformSemantics			pointer to uniform semantic data array
	@Input				nSemantics					number of uniform semantics
	@Output				pReturnError				error string
	@Returns			EError					PVR_SUCCESS if succeeded
	@Description		Builds the uniform table from the semantics.
	*****************************************************************************/
	unsigned int BuildUniformTable(
		SPFXUniform					** const ppsUniforms,
		unsigned int					* const pnUniformCount,
		unsigned int					* const pnUnknownUniformCount,
		const SPFXUniformSemantic	* const psUniformSemantics,
		const unsigned int				nSemantics,
		string					*pReturnError);

	/*!***************************************************************************
	@Function			GetTextureArray
	@Output				nCount					number of textures
	@Returns			SPFXTexture*		pointer to the texture data array
	@Description		Gets the texture data array.
	*****************************************************************************/
	const SPFXTexture *GetTextureArray(unsigned int &nCount) const;

	/*!***************************************************************************
	@Function			SetTexture
	@Input				nIdx				texture number
	@Input				ui					opengl texture handle
	@Input				u32flags			texture flags
	@Description		Sets the textrue and applys the filtering.
	*****************************************************************************/
	void SetTexture(const unsigned int nIdx, const GLuint ui, const unsigned int u32flags=0);

	/*!***************************************************************************
	@Function			SetDefaultSemanticValue
	@Input				pszName				name of uniform
	@Input				psDefaultValue      pointer to default value
	@Description		Sets the dafault value for the uniform semantic.
	*****************************************************************************/
	void SetDefaultUniformValue(const char *const pszName, const SSemanticDefaultData *psDefaultValue);

};

#endif /* _PFXPARSERAPI_H_ */

/*****************************************************************************
 End of file (PFXParserAPI.h)
*****************************************************************************/
