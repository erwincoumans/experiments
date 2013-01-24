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

#include <TargetConditionals.h>
#include <Availability.h>

#if __IPHONE_OS_VERSION_MAX_ALLOWED >= 30000
#import <OpenGLES/ES2/gl.h>
#import <OpenGLES/ES2/glext.h>
#else
#import <OpenGLES/ES1/gl.h>
#import <OpenGLES/ES1/glext.h>
#endif


#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//#define FORCE_NO_LOGO

#include "Mathematics.h"
#include "Macros.h"

#include "DisplayText.h"

#if __IPHONE_OS_VERSION_MAX_ALLOWED >= 30000
extern int __OPENGLES_VERSION;
#endif

struct SDisplayTextAPI
{
	GLuint						uTexture[5];
	GLuint						uTexturePVRLogo;
	GLuint						uTextureIGDKLogo;
};

//
// Deallocate the memory allocated in DisplayTextSetTextures(...)
//
void CDisplayText::ReleaseTextures()
{
#if !defined (DISABLE_DISPLAYTEXT)

	/* Only release textures if they've been allocated */
	if (!m_bTexturesSet) return;

	/* Release IndexBuffer */
	free(m_pwFacesFont);
	free(m_pPrint3dVtx);

	/* Delete textures */
	glDeleteTextures(5, m_pAPI->uTexture);
	glDeleteTextures(1, &m_pAPI->uTexturePVRLogo);
	glDeleteTextures(1, &m_pAPI->uTextureIGDKLogo);

	m_bTexturesSet = false;

	free(m_pVtxCache);

	APIRelease();

#endif
}

//
// Flushes all the print text commands
//
int CDisplayText::Flush()
{
#if !defined (DISABLE_DISPLAYTEXT)

	int		nTris, nVtx, nVtxBase, nTrisTot;

	_ASSERT((m_nVtxCache % 4) == 0);
	_ASSERT(m_nVtxCache <= m_nVtxCacheMax);
	
	
	/* Save render states */
	APIRenderStates(0);

	/* Set font texture */
	glBindTexture(GL_TEXTURE_2D, m_pAPI->uTexture[0]);	

	/* Set blending mode */
	//glEnable(GL_BLEND);

	nTrisTot = m_nVtxCache >> 1;

	/*
		Render the text then. Might need several submissions.
	*/
	nVtxBase = 0;
	while(m_nVtxCache)
	{
		nVtx	= _MIN(m_nVtxCache, 0xFFFC);
		nTris	= nVtx >> 1;

		_ASSERT(nTris <= (DISPLAYTEXT_MAX_RENDERABLE_LETTERS*2));
		_ASSERT((nVtx % 4) == 0);

#if __IPHONE_OS_VERSION_MAX_ALLOWED >= 30000
		
		if( __OPENGLES_VERSION >= 2 )
		{
			// Bind the VBO so we can fill it with data
			glBindBuffer(GL_ARRAY_BUFFER, 0);
			
			glEnableVertexAttribArray(0);
			glVertexAttribPointer(0, 3, VERTTYPEENUM, GL_FALSE, sizeof(SDisplayTextAPIVertex), &m_pVtxCache[nVtxBase].sx);	

			glEnableVertexAttribArray(1);
			glVertexAttribPointer(1, 4, VERTTYPEENUM, GL_FALSE, sizeof(SDisplayTextAPIVertex), &m_pVtxCache[nVtxBase].r);	
			
			glEnableVertexAttribArray(2);
			glVertexAttribPointer(2, 2, VERTTYPEENUM, GL_FALSE, sizeof(SDisplayTextAPIVertex), &m_pVtxCache[nVtxBase].tu);

			if (glGetError())
			{
				_RPT0(_CRT_WARN,"Error while binding bufer for CDisplayText::Flush\n");
			}
		}
		else 
#endif
		{		
			/* Draw triangles */
			glVertexPointer(3,		VERTTYPEENUM,		sizeof(SDisplayTextAPIVertex), &m_pVtxCache[nVtxBase].sx);
			glColorPointer(4,		VERTTYPEENUM,	sizeof(SDisplayTextAPIVertex), &m_pVtxCache[nVtxBase].r);
			glTexCoordPointer(2,	VERTTYPEENUM,		sizeof(SDisplayTextAPIVertex), &m_pVtxCache[nVtxBase].tu);

		}
		
		glDrawElements(GL_TRIANGLES, nTris * 3, GL_UNSIGNED_SHORT, m_pwFacesFont);
		//glDrawArrays(GL_TRIANGLES,  0, 3);
		
		if (glGetError())
		{
			_RPT0(_CRT_WARN,"glDrawElements(GL_TRIANGLES, (VertexCount/2)*3, GL_UNSIGNED_SHORT, m_pFacesFont); failed\n");
		}
		
		nVtxBase		+= nVtx;
		m_nVtxCache	-= nVtx;
	}
	


	{
		/* Draw a logo if requested */
	#if defined(FORCE_NO_LOGO)
		/* Do nothing */

	#elif defined(FORCE_PVR_LOGO)
		APIDrawLogo(eDisplayTextLogoPVR, 1);	/* PVR to the right */

	#elif defined(FORCE_IMG_LOGO)
		APIDrawLogo(eDisplayTextLogoIMG, 1);	/* IMG to the right */

	#elif defined(FORCE_ALL_LOGOS)
		APIDrawLogo(eDisplayTextLogoIMG, -1); /* IMG to the left */
		APIDrawLogo(eDisplayTextLogoPVR, 1);	/* PVR to the right */

	#else
		/* User selected logos */
		switch (m_uLogoToDisplay)
		{
			case eDisplayTextLogoNone:
				break;
			default:
			case eDisplayTextLogoPVR:
				APIDrawLogo(eDisplayTextLogoPVR, 1);	/* PVR to the right */
				break;
			case eDisplayTextLogoIMG:
				APIDrawLogo(eDisplayTextLogoIMG, 1);	/* IMG to the right */
				break;
			case (eDisplayTextLogoPVR | eDisplayTextLogoIMG):
				APIDrawLogo(eDisplayTextLogoIMG, -1); /* IMG to the left */
				APIDrawLogo(eDisplayTextLogoPVR, 1);	/* PVR to the right */
				break;
		}
	#endif
	}

	/* Restore render states */
	APIRenderStates(1);
	
	return nTrisTot;

#else
	return 0;
#endif
}



//
// Initialization and texture upload. Should be called only once
// for a given context.
//
bool CDisplayText::APIInit()
{
    m_pAPI = new SDisplayTextAPI;

	if(!m_pAPI)
		return false;
	return true;
}

//
// Deinitialization.
//
void CDisplayText::APIRelease()
{
	delete m_pAPI;
	m_pAPI = 0;
}

//
// Initialization and texture upload. Should be called only once
// for a given context.
//
bool CDisplayText::APIUpLoadIcons(
	const unsigned long * const pPVR,
	const unsigned long * const pIMG)
{
	/* Load Icon textures */
	/* PVR Icon */
	glGenTextures(1, &m_pAPI->uTexturePVRLogo);
	glBindTexture(GL_TEXTURE_2D, m_pAPI->uTexturePVRLogo);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 128, 128, 0, GL_RGB, GL_UNSIGNED_SHORT_5_6_5, pPVR + (pPVR[0] / sizeof(unsigned long)));

	/* IMG Icon */
	glGenTextures(1, &m_pAPI->uTextureIGDKLogo);
	glBindTexture(GL_TEXTURE_2D, m_pAPI->uTextureIGDKLogo);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 128, 128, 0, GL_RGB, GL_UNSIGNED_SHORT_5_6_5, pIMG + (pIMG[0] / sizeof(unsigned long)));
	return true;
}

//
// true if succesful, false otherwise.
// Reads texture data from *.dat and loads it in
// video memory.
//
bool CDisplayText::APIUpLoad4444(unsigned int dwTexID, unsigned char *pSource, unsigned int nSize, unsigned int nMode)
{
	int				i, j;
	int				x=256, y=256;
	unsigned short	R, G, B, A;
	unsigned short	*p8888,  *pDestByte;
	unsigned char   *pSrcByte;

	/* Only square textures */
	x = nSize;
	y = nSize;

	glGenTextures(1, &m_pAPI->uTexture[dwTexID]);

	/* Load texture from data */

	/* Format is 4444-packed, expand it into 8888 */
	if (nMode==0)
	{
		/* Allocate temporary memory */
		p8888 = (unsigned short *) malloc(nSize * nSize * sizeof(unsigned short));
		memset(p8888, 0, nSize * nSize * sizeof(unsigned short));
		pDestByte = p8888;

		/* Set source pointer (after offset of 16) */
		pSrcByte = &pSource[16];

		/* Transfer data */
		for (i=0; i<y; i++)
		{
			for (j=0; j<x; j++)
			{
				/* Get all 4 colour channels (invert A) */
				G =   (*pSrcByte) & 0xF0;
				R = ( (*pSrcByte++) & 0x0F ) << 4;
				A =   (*pSrcByte) ^ 0xF0;
				B = ( (*pSrcByte++) & 0x0F ) << 4;

				/* Set them in 8888 data */
				*pDestByte++ = ((R&0xF0)<<8) | ((G&0xF0)<<4) | (B&0xF0) | (A&0xF0)>>4;
			}
		}
	}
	else
	{
		/* Set source pointer */
		pSrcByte = pSource;

		/* Allocate temporary memory */
		p8888 = (unsigned short *) malloc (nSize*nSize*sizeof(unsigned short));
		memset(p8888, 0, nSize*nSize*sizeof(unsigned short));

		if (!p8888)
		{
			_RPT0(_CRT_WARN,"Not enough memory!\n");
			return false;
		}

		/* Set destination pointer */
		pDestByte = p8888;

		/* Transfer data */
		for (i=0; i<y; i++)
		{
			for (j=0; j<x; j++)
			{
				/* Get alpha channel */
				A = *pSrcByte++;

				/* Set them in 8888 data */
				R = 255;
				G = 255;
				B = 255;

				/* Set them in 8888 data */
				*pDestByte++ = ((R&0xF0)<<8) | ((G&0xF0)<<4) | (B&0xF0) | (A&0xF0)>>4;
			}
		}
	}

	/* Bind texture */
	glBindTexture(GL_TEXTURE_2D, m_pAPI->uTexture[dwTexID]);

	/* Default settings: bilinear */
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

	/* Now load texture */
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, x, y, 0, GL_RGBA, GL_UNSIGNED_SHORT_4_4_4_4, p8888);

	if (glGetError())
	{
		_RPT0(_CRT_WARN,"glTexImage2D() failed\n");
		free(p8888);
		return false;
	}

	/* Destroy temporary data */
	free(p8888);

	/* Return status : OK */
	return true;
}


void CDisplayText::DrawBackgroundWindowUP(unsigned int dwWin, SDisplayTextAPIVertex *pVtx, const bool bIsOp, const bool bBorder)
{
	const unsigned short c_pwFacesWindow[] =
	{
		0,1,2, 2,1,3, 2,3,4, 4,3,5, 4,5,6, 6,5,7, 5,8,7, 7,8,9, 8,10,9, 9,10,11, 8,12,10, 8,13,12,
		13,14,12, 13,15,14, 13,3,15, 1,15,3, 3,13,5, 5,13,8
	};

	/* Set the texture (with or without border) */
	if(!bBorder)
		glBindTexture(GL_TEXTURE_2D, m_pAPI->uTexture[2 + (bIsOp*2)]);
	else
		glBindTexture(GL_TEXTURE_2D, m_pAPI->uTexture[1 + (bIsOp*2)]);

	/* Is window opaque ? */
	if(bIsOp)
	{
		glDisable(GL_BLEND);
	}
	else
	{
		/* Set blending properties */
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	}

	/* Set pointers */
	glVertexPointer(3,		VERTTYPEENUM,		sizeof(SDisplayTextAPIVertex), &pVtx[0].sx);
	glColorPointer(4,		VERTTYPEENUM,	sizeof(SDisplayTextAPIVertex), &pVtx[0].r);
	glTexCoordPointer(2,	VERTTYPEENUM,		sizeof(SDisplayTextAPIVertex), &pVtx[0].tu);

	/* Draw triangles */
	
	glDrawElements(GL_TRIANGLES, 18*3, GL_UNSIGNED_SHORT, c_pwFacesWindow);
	if (glGetError())
	{
		_RPT0(_CRT_WARN,"glDrawElements(GL_TRIANGLES, 18*3, GL_UNSIGNED_SHORT, pFaces); failed\n");
	}

	/* Restore render states (need to be translucent to draw the text) */
}

//
// Stores, writes and restores Render States
//
void CDisplayText::APIRenderStates(int nAction)
{
	static GLint		iMatrixMode, iFrontFace, iCullFaceMode, iDestBlend, iSrcBlend;
	static GLboolean	bLighting, bCullFace, bFog, bDepthTest, bBlend, bVertexPointerEnabled, bColorPointerEnabled, bTexCoorPointerEnabled ; //, bVertexProgram;
   static GLboolean  bTextureEnabled0, bTextureEnabled1;
	MATRIX Matrix;
	int i;

	/* Saving or restoring states ? */
	switch (nAction)
	{
	case 0:
		/* Get previous render states */
		/* Save all attributes */
		/*glPushAttrib(GL_ALL_ATTRIB_BITS);*/

		/* Client states */
		//glGetbooleanv(GL_VERTEX_ARRAY,		&bVertexPointerEnabled);
		//glGetbooleanv(GL_COLOR_ARRAY,			&bColorPointerEnabled);
		//glGetbooleanv(GL_TEXTURE_COORD_ARRAY,	&bTexCoorPointerEnabled);
#if __IPHONE_OS_VERSION_MAX_ALLOWED >= 30000
		if( __OPENGLES_VERSION >= 2 ) {
		}
		else 
#endif
		{		
			bVertexPointerEnabled = glIsEnabled(GL_VERTEX_ARRAY);
			bColorPointerEnabled = glIsEnabled(GL_COLOR_ARRAY);
			bTexCoorPointerEnabled = glIsEnabled(GL_TEXTURE_COORD_ARRAY);
			bLighting = glIsEnabled(GL_LIGHTING);
			bFog = glIsEnabled(GL_FOG);
			// save texture unit state
			glActiveTexture(GL_TEXTURE0);
			bTextureEnabled0 = glIsEnabled(GL_TEXTURE_2D);
			
			glActiveTexture(GL_TEXTURE1);
			bTextureEnabled1 = glIsEnabled(GL_TEXTURE_2D);
		}

		bCullFace = glIsEnabled(GL_CULL_FACE);
		bDepthTest = glIsEnabled(GL_DEPTH_TEST);
//		bVertexProgram = glIsEnabled(GL_IMG_vertex_program);
		bBlend = glIsEnabled(GL_BLEND);
		glGetIntegerv(GL_FRONT_FACE, &iFrontFace);
		glGetIntegerv(GL_CULL_FACE_MODE, &iCullFaceMode);
		glGetIntegerv(GL_BLEND_DST, &iDestBlend);
		glGetIntegerv(GL_BLEND_SRC, &iSrcBlend);
      
		
#if __IPHONE_OS_VERSION_MAX_ALLOWED >= 30000
		if( __OPENGLES_VERSION >= 2 ) {
			//glBindBuffer( GL_ARRAY_BUFFER, 0 );
			glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, 0 );
		}
		else 
#endif
		{
			/* Save matrices */
			glGetIntegerv(GL_MATRIX_MODE, &iMatrixMode);
			
			glMatrixMode(GL_MODELVIEW);
			glPushMatrix();
			glMatrixMode(GL_PROJECTION);
			glPushMatrix();
		}		
			
		/******************************
		** SET DisplayText RENDER STATES **
		******************************/

		/* Get viewport dimensions */
		/*glGetFloatv(GL_VIEWPORT, fViewport);*/

		/* Set matrix with viewport dimensions */
		for(i=0; i<16; i++)
		{
			Matrix.f[i]=0;
		}
		Matrix.f[0] =	f2vt(2.0f/(m_fScreenScale[0]*WindowWidth));
		Matrix.f[5] =	f2vt(-2.0f/(m_fScreenScale[1]*WindowHeight));

		
		Matrix.f[10] = f2vt(1.0f);
		Matrix.f[12] = f2vt(-1.0f);
		Matrix.f[13] = f2vt(1.0f);
		Matrix.f[15] = f2vt(1.0f);
			
			
#if __IPHONE_OS_VERSION_MAX_ALLOWED >= 30000
		if( __OPENGLES_VERSION >= 2 ) {
			glUseProgram(uiProgramObject);
			glUniformMatrix4fv( PMVMatrixHandle, 1, GL_FALSE, Matrix.f);
		}
		else 
#endif
		{			
			/* Set matrix mode so that screen coordinates can be specified */
			glMatrixMode(GL_PROJECTION);
			glLoadIdentity();

			glMatrixMode(GL_MODELVIEW);
			
			glLoadMatrixf(Matrix.f);
			glDisable(GL_LIGHTING);
		}
					
			
		/* Culling */
		glEnable(GL_CULL_FACE);
		glFrontFace(GL_CW);
		glCullFace(GL_FRONT);

#if __IPHONE_OS_VERSION_MAX_ALLOWED >= 30000
		if( __OPENGLES_VERSION >= 2 ) {
			glActiveTexture(GL_TEXTURE0);
			glUniform1i( TextureHandle, 0 );
		}
		else 		
#endif
		{	
			/* Set client states */
			glEnableClientState(GL_VERTEX_ARRAY);
			glEnableClientState(GL_COLOR_ARRAY);
			glClientActiveTexture(GL_TEXTURE0);
			glEnableClientState(GL_TEXTURE_COORD_ARRAY);
			
			/* texture 	*/
			glActiveTexture(GL_TEXTURE1);
			glDisable(GL_TEXTURE_2D);
			glActiveTexture(GL_TEXTURE0);
			glEnable(GL_TEXTURE_2D);
		}

#if __IPHONE_OS_VERSION_MAX_ALLOWED >= 30000
		if( __OPENGLES_VERSION >= 2 ) {
		}
		else 
#endif
		{			
			glTexEnvf(GL_TEXTURE_ENV,GL_TEXTURE_ENV_MODE,GL_MODULATE);
			/* Disable fog */
			glDisable(GL_FOG);
		}

		/* Blending mode */
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

		/* Set Z compare properties */
		glDisable(GL_DEPTH_TEST);

		// Disable vertex program
//		glDisable(GL_IMG_vertex_program);			
			
		break;

	case 1:
#if __IPHONE_OS_VERSION_MAX_ALLOWED >= 30000
		if( __OPENGLES_VERSION >= 2 ) {
		}
		else 
#endif
		{				
			/* Restore render states */
			if (!bVertexPointerEnabled)		
				glDisableClientState(GL_VERTEX_ARRAY);
			else 
				glEnableClientState(GL_VERTEX_ARRAY);
			if (!bColorPointerEnabled)		
				glDisableClientState(GL_COLOR_ARRAY);
			else
				glEnableClientState(GL_COLOR_ARRAY);
			if (!bTexCoorPointerEnabled)		
				glDisableClientState(GL_TEXTURE_COORD_ARRAY);
			else
				glEnableClientState(GL_TEXTURE_COORD_ARRAY);
		}

#if __IPHONE_OS_VERSION_MAX_ALLOWED >= 30000
		if( __OPENGLES_VERSION >= 2 ) {
		}
		else 
#endif
		{
			/* Restore matrix mode & matrix */
			glMatrixMode(GL_PROJECTION);
			glPopMatrix();
			glMatrixMode(GL_MODELVIEW);
			glPopMatrix();

			glMatrixMode(iMatrixMode);
			if(bLighting)		glEnable(GL_LIGHTING);
			if(bFog)			glEnable(GL_FOG);

			// restore texture states
			glActiveTexture(GL_TEXTURE1);
			bTextureEnabled1 ? glEnable(GL_TEXTURE_2D) : glDisable(GL_TEXTURE_2D);
			glActiveTexture(GL_TEXTURE0);
			bTextureEnabled0 ? glEnable(GL_TEXTURE_2D) : glDisable(GL_TEXTURE_2D);
		}
			

		// Restore some values
		if(!bCullFace)		glDisable(GL_CULL_FACE);
		if(bDepthTest)		glEnable(GL_DEPTH_TEST);
//		if(bVertexProgram)	glEnable(GL_IMG_vertex_program);
			
		glFrontFace(iFrontFace);
		glCullFace(iCullFaceMode);

		glBlendFunc(iSrcBlend, iDestBlend);
		if(bBlend == 0) glDisable(GL_BLEND);



		break;
	}
}

//
// nPos = -1 to the left
// nPos = +1 to the right
//
#define LOGO_SIZE 0.3f
#define LOGO_SHIFT 0.08f

void CDisplayText::APIDrawLogo(unsigned int uLogoToDisplay, int nPos)
{
	// 480 by 320 is the resolution, so the logos need to be adjusted to this
	static VERTTYPE	VerticesRight[] = {
			f2vt(1.0f-LOGO_SHIFT-(LOGO_SIZE * (WindowHeight/WindowWidth))), f2vt(-1.0f+(LOGO_SIZE)+LOGO_SHIFT)	, f2vt(0.5f),
			f2vt(1.0f-LOGO_SHIFT-(LOGO_SIZE * (WindowHeight/WindowWidth))), f2vt(-1.0f+LOGO_SHIFT)				, f2vt(0.5f),
			f2vt(1.0f-LOGO_SHIFT)			, f2vt(-1.0f+(LOGO_SIZE)+LOGO_SHIFT)					, f2vt(0.5f),
	 		f2vt(1.0f-LOGO_SHIFT)	 		, f2vt(-1.0f+LOGO_SHIFT)								, f2vt(0.5f)
		};

	static VERTTYPE	VerticesLeft[] = {
			f2vt(-1.0f+LOGO_SHIFT)			, f2vt(-1.0f+(LOGO_SIZE)+LOGO_SHIFT)					, f2vt(0.5f),
			f2vt(-1.0f+LOGO_SHIFT)			, f2vt(-1.0f+LOGO_SHIFT)								, f2vt(0.5f),
			f2vt(-1.0f+LOGO_SHIFT+(LOGO_SIZE * (WindowHeight/WindowWidth))), f2vt(-1.0f+(LOGO_SIZE)+LOGO_SHIFT)	, f2vt(0.5f),
	 		f2vt(-1.0f+LOGO_SHIFT+(LOGO_SIZE * (WindowHeight/WindowWidth))), f2vt(-1.0f+LOGO_SHIFT)			, f2vt(0.5f)
		};

	static VERTTYPE	Colours[] = {
			f2vt(1.0f), f2vt(1.0f), f2vt(1.0f), f2vt(0.75f),
			f2vt(1.0f), f2vt(1.0f), f2vt(1.0f), f2vt(0.75f),
			f2vt(1.0f), f2vt(1.0f), f2vt(1.0f), f2vt(0.75f),
	 		f2vt(1.0f), f2vt(1.0f), f2vt(1.0f), f2vt(0.75f)
		};

	static VERTTYPE	UVs[] = {
			f2vt(0.0f), f2vt(0.0f),
			f2vt(0.0f), f2vt(1.0f),
			f2vt(1.0f), f2vt(0.0f),
	 		f2vt(1.0f), f2vt(1.0f)
		};

	VERTTYPE *pVertices = ( (VERTTYPE*)&VerticesRight );
	VERTTYPE *pColours  = ( (VERTTYPE*)&Colours );
	VERTTYPE *pUV       = ( (VERTTYPE*)&UVs );
	GLuint	tex;

	switch(uLogoToDisplay)
	{
	case eDisplayTextLogoIMG:
		tex = m_pAPI->uTextureIGDKLogo;
		break;
	default:
		tex = m_pAPI->uTexturePVRLogo;
		break;
	}

	/* Left hand side of the screen */
	if (nPos == -1)
	{
		pVertices = VerticesLeft;
	}

#if __IPHONE_OS_VERSION_MAX_ALLOWED >= 30000
	if( __OPENGLES_VERSION >= 2 ) {
		MATRIX mx;
		MatrixIdentity(mx);
		if(bScreenRotate)
		{
			MatrixRotationZ( mx, 90.0 * PIf /180.0f );
		}
		glUseProgram(uiProgramObject);
		glUniformMatrix4fv( PMVMatrixHandle, 1, GL_FALSE, mx.f);
		
		// Render states
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, tex);
	}
	else 
#endif		
//#else
	{	
		//Matrices
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();

		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		if(bScreenRotate)
		{
			glRotatef(f2vt(-90), f2vt(0), f2vt(0), f2vt(1));
		}
		glActiveTexture(GL_TEXTURE0);
		glEnable(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, tex);
	}
//#endif
	


	glEnable (GL_BLEND);
	glBlendFunc (GL_ZERO, GL_SRC_COLOR);

#if __IPHONE_OS_VERSION_MAX_ALLOWED >= 30000
	
	if( __OPENGLES_VERSION >= 2 )
	{
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, VERTTYPEENUM, GL_FALSE, 0, pVertices);	
		
		//glEnableVertexAttribArray(1);
		//glVertexAttribPointer(1, 4, GL_UNSIGNED_BYTE, GL_FALSE, sizeof(SDisplayTextAPIVertex), &m_pVtxCache[nVtxBase].color);	
		
		glEnableVertexAttribArray(2);
		glVertexAttribPointer(2, 2, VERTTYPEENUM, GL_FALSE, 0, pUV);
		
		glUniform1i( TextureHandle, 0 );			
		
		glDrawArrays(GL_TRIANGLE_STRIP,0,4);
	}
	else 
#endif	// Vertices
	{
		glEnableClientState(GL_VERTEX_ARRAY);
		glVertexPointer(3,VERTTYPEENUM,0,pVertices);

		glEnableClientState(GL_COLOR_ARRAY);
		glColorPointer(4,VERTTYPEENUM,0,pColours);

		glClientActiveTexture(GL_TEXTURE0);
		glEnableClientState(GL_TEXTURE_COORD_ARRAY);
		glTexCoordPointer(2,VERTTYPEENUM,0,pUV);
		
		glDrawArrays(GL_TRIANGLE_STRIP,0,4);
		
		glDisableClientState(GL_VERTEX_ARRAY);
		glDisableClientState(GL_COLOR_ARRAY);
		glClientActiveTexture(GL_TEXTURE0);
		glDisableClientState(GL_TEXTURE_COORD_ARRAY);
	}



	// Restore render states
	glDisable (GL_BLEND);
}
