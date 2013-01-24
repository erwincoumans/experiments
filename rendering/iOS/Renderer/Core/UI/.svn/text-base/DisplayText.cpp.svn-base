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
#include <stdarg.h>
#include <stdio.h>
#include <string.h>


#include "DeviceType.h"
#include "Mathematics.h"
#include "Geometry.h"
#include "Macros.h"
#include "MemoryManager.h"
#if __IPHONE_OS_VERSION_MAX_ALLOWED >= 30000
#include "Shader.h"
#endif

#include "DisplayText.h"
#include "DisplayTextdat.h"		// texture data

#define MAX_LETTERS				(5120)
#define MIN_CACHED_VTX			(0x1000)
#define MAX_CACHED_VTX			(0x00100000)
#define LINES_SPACING			(29.0f)

#define DisplayText_WIN_EXIST	1
#define DisplayText_WIN_ACTIVE	2
#define DisplayText_WIN_TITLE	4
#define DisplayText_WIN_STATIC	8
#define DisplayText_FULL_OPAQUE	16
#define DisplayText_FULL_TRANS	32
#define DisplayText_ADJUST_SIZE	64
#define DisplayText_NO_BORDER	128

#if __IPHONE_OS_VERSION_MAX_ALLOWED >= 30000
extern int __OPENGLES_VERSION;
#endif

float WindowWidth = 320.0f;
float WindowHeight = 480.0f;

#if __IPHONE_OS_VERSION_MAX_ALLOWED >= 30000
// This ties in with the shader attribute to link to openGL, see pszVertShader.
static const char* pszAttribs[] = { "myVertex", "myColor", "myTexCoord" };

	

static char vsh[] = "\
attribute highp vec4	myVertex;											\
attribute highp vec4	myColor;											\
attribute highp vec2	myTexCoord;											\
																			\
uniform mediump mat4	myPMVMatrix;										\
																			\
varying mediump vec2	v_textureCoord;										\
varying mediump vec4	v_color;											\
																			\
void main(void)																\
{																			\
	gl_Position = myPMVMatrix * myVertex;									\
	v_textureCoord = myTexCoord;											\
	v_color = myColor;														\
}																			\
";

static char fsh[] = "\
varying mediump vec2	v_textureCoord;										\
varying mediump vec4	v_color;											\
uniform sampler2D		s_texture;											\
																			\
void main(void)																\
{																			\
	gl_FragColor = texture2D(s_texture, v_textureCoord) * v_color;			\
}																			\
";


#endif

//DEFINE_HEAP(CDisplayText, "UI");

CDisplayText::CDisplayText()
{
#if !defined(DISABLE_DISPLAYTEXT)

	// Initialise all variables
	memset(this, 0, sizeof(*this));
	
#if __IPHONE_OS_VERSION_MAX_ALLOWED >= 30000
	if( __OPENGLES_VERSION >= 2 ) {
		if(ShaderLoadSourceFromMemory( fsh, GL_FRAGMENT_SHADER, &uiFragShader) == 0)
			printf("Loading the fragment shader fails");
		if(ShaderLoadSourceFromMemory( vsh, GL_VERTEX_SHADER, &uiVertShader) == 0)
			printf("Loading the vertex shader fails");
		
		CreateProgram(&uiProgramObject, uiVertShader, uiFragShader, pszAttribs, 3);
		
		// First gets the location of that variable in the shader using its name
		PMVMatrixHandle = glGetUniformLocation(uiProgramObject, "myPMVMatrix");
		TextureHandle = glGetUniformLocation(uiProgramObject, "s_texture");			
	}
#endif

#endif
}


CDisplayText::~CDisplayText()
{
#if !defined (DISABLE_DISPLAYTEXT)
#if __IPHONE_OS_VERSION_MAX_ALLOWED >= 30000	
	if( __OPENGLES_VERSION >= 2 )
	{
		// Frees the OpenGL handles for the program and the 2 shaders
		glDeleteProgram(uiProgramObject);
		glDeleteShader(uiFragShader);
		glDeleteShader(uiVertShader);
	}
#endif
	APIRelease();
#endif
}

//
// pContext			Context
// dwScreenX		Screen resolution along X
// dwScreenY		Screen resolution along Y
// true or false
// Initialization and texture upload. Should be called only once
// for a given context.
//
bool CDisplayText::SetTextures(
	const unsigned int	dwScreenX,
	const unsigned int	dwScreenY,
	bool			bRotate) 
{
#if !defined (DISABLE_DISPLAYTEXT)

	int				i;
	bool			bStatus;

	bScreenRotate = bRotate; 
	//bScreenRotate = false;
	if( bRotate ) {
#if __IPHONE_OS_VERSION_MAX_ALLOWED >= 30200
		if (GetDeviceType()==IPAD_DEVICE)
		{
			WindowWidth = 1024;
			WindowHeight = 768;
		}
		else 
#endif
		{
			WindowWidth = 480;
			WindowHeight = 320;
		}

	}
	else {
#if __IPHONE_OS_VERSION_MAX_ALLOWED >= 30200
		if (GetDeviceType()==IPAD_DEVICE)
		{
			WindowWidth = 768;
			WindowHeight = 1024;
		}
#endif
	}
	/* Set the aspect ratio, so we can change it without updating textures or anything else */
	m_fScreenScale[0] = (float)dwScreenX/WindowWidth;
	m_fScreenScale[1] = (float)dwScreenY/WindowHeight;

	/* Check whether textures are already set up just in case */
	if (m_bTexturesSet) return true;

	if(!APIInit())
		return false;

	/*
		This is the window background texture
		Type 0 because the data comes in TexTool rectangular format.
	*/
	bStatus = APIUpLoad4444(1, (unsigned char *)WindowBackground, 16, 0);
	if (!bStatus) return false;

	bStatus = APIUpLoad4444(2, (unsigned char *)WindowPlainBackground, 16, 0);
	if (!bStatus) return false;

	bStatus = APIUpLoad4444(3, (unsigned char *)WindowBackgroundOp, 16, 0);
	if (!bStatus) return false;

	bStatus = APIUpLoad4444(4, (unsigned char *)WindowPlainBackgroundOp, 16, 0);
	if (!bStatus) return false;

	/*
		This is the texture with the fonts.
		Type 1 because there is only alpha component (RGB are white).
	*/
	bStatus = APIUpLoad4444(0, (unsigned char *)DisplayTextABC_Pixels, 256, 1);
	if (!bStatus) return false;

	/* INDEX BUFFERS */
	m_pwFacesFont = (unsigned short*) malloc(DISPLAYTEXT_MAX_RENDERABLE_LETTERS * 2 * 3 * sizeof(unsigned short));
	memset(m_pwFacesFont, 0, DISPLAYTEXT_MAX_RENDERABLE_LETTERS * 2 * 3 * sizeof(unsigned short));	

	bStatus = APIUpLoadIcons(DisplayTextPVRLogo, iGDKLogo);
	if (!bStatus) return false;

	/* Vertex indices for letters */
	for (i=0; i < DISPLAYTEXT_MAX_RENDERABLE_LETTERS; i++)
	{
		m_pwFacesFont[i*6+0] = 0+i*4;
		m_pwFacesFont[i*6+1] = 3+i*4;
		m_pwFacesFont[i*6+2] = 1+i*4;

		m_pwFacesFont[i*6+3] = 3+i*4;
		m_pwFacesFont[i*6+4] = 0+i*4;
		m_pwFacesFont[i*6+5] = 2+i*4;
	}

	m_nVtxCacheMax = MIN_CACHED_VTX;
	m_pVtxCache = (SDisplayTextAPIVertex*)malloc(m_nVtxCacheMax * sizeof(*m_pVtxCache));
	memset(m_pVtxCache, 0, m_nVtxCacheMax * sizeof(*m_pVtxCache));	
	m_nVtxCache = 0;

	/* Everything is OK */
	m_bTexturesSet = true;

	/* set all windows for an update */
	for (i=0; i<DISPLAYTEXT_MAX_WINDOWS; i++)
		m_pWin[i].bNeedUpdated = true;

	/* Return OK */
	return true;

#else
	return 1;
#endif
}

//
// fPosX		Position of the text along X
// fPosY		Position of the text along Y
// fScale		Scale of the text
// Colour		Colour of the text
// pszFormat	Format string for the text
// Display 3D text on screen.
// No window needs to be allocated to use this function.
// However, DisplayTextSetTextures(...) must have been called
// beforehand.
// This function accepts formatting in the printf way.
//
void CDisplayText::DisplayText(float fPosX, float fPosY, const float fScale, unsigned int Colour, const char * const pszFormat, ...)
{
#if !defined (DISABLE_DISPLAYTEXT)

	va_list				args;
	static char			Text[MAX_LETTERS+1], sPreviousString[MAX_LETTERS+1];
	static float		XPosPrev, YPosPrev, fScalePrev;
	static unsigned int	ColourPrev;
	static unsigned int	nVertices;

	/* No textures! so... no window */
	if (!m_bTexturesSet)
	{
		_RPT0(_CRT_WARN,"DisplayTextDisplayWindow : You must call DisplayTextSetTextures()\nbefore using this function!!!\n");
		return;
	}

	/* Reading the arguments to create our Text string */
	va_start(args, pszFormat);
	vsprintf(Text, pszFormat, args);		// Could use _vsnprintf but e.g. LinuxVP does not support it
	va_end(args);

	/* nothing to be drawn */
	if(*Text == 0)
		return;

	/* Adjust input parameters */
	fPosX *= WindowWidth/100.0f;
	fPosY *= WindowHeight/100.0f;

	/* We check if the string has been changed since last time */
	if(
		strcmp (sPreviousString, Text) != 0 ||
		fPosX != XPosPrev ||
		fPosY != YPosPrev ||
		fScale != fScalePrev ||
		Colour != ColourPrev ||
		m_pPrint3dVtx == NULL)
	{
		/* copy strings */
		strcpy (sPreviousString, Text);
		XPosPrev = fPosX;
		YPosPrev = fPosY;
		fScalePrev = fScale;
		ColourPrev = Colour;

		/* Create Vertex Buffer (only if it doesn't exist) */
		if(m_pPrint3dVtx==0)
		{
			m_pPrint3dVtx = (SDisplayTextAPIVertex*) malloc(MAX_LETTERS*4*sizeof(SDisplayTextAPIVertex));
			memset(m_pPrint3dVtx, 0, MAX_LETTERS*4*sizeof(SDisplayTextAPIVertex));
		}

		/* Fill up our buffer */
		nVertices = UpdateLine(0, 0.0f, fPosX, fPosY, fScale, Colour, Text, m_pPrint3dVtx);
	}

	// Draw the text
	DrawLineUP(m_pPrint3dVtx, nVertices);

#endif
}
// 
// sTitle				Title to display
// sDescription		Description to display
// uDisplayLogo		1 = Display the logo
// Creates a default title with predefined position and colours.
// It displays as well company logos when requested:
// 0 = No logo
// 1 = PowerVR logo
// 2 = iGDK logo
//
void CDisplayText::DisplayDefaultTitle(const char * const pszTitle, const char * const pszDescription, const unsigned int uDisplayLogo)
{
#if !defined (DISABLE_DISPLAYTEXT)

	/* Display Title
	 */
	if(pszTitle)
	{   // please note the scale value is hard-coded here
		DisplayText(0.0f, 1.0f, 0.6f,  RGBA(255, 255, 0, 255), pszTitle);
	}

	/* Display Description
	 */
	if(pszDescription)
	{
		DisplayText(0.0f, 6.0f, 0.4f,  RGBA(255, 255, 255, 255), pszDescription);
	}

	m_uLogoToDisplay = uDisplayLogo;

#endif
}

//
// fPosX					Position X for the new window
// fPosY					Position Y for the new window
// nXSize_LettersPerLine
// sTitle					Title of the window
// sBody					Body text of the window
// Window handle
// Creates a default window.
// If Title is NULL the main body will have just one line
// (for InfoWin).
//
unsigned int CDisplayText::CreateDefaultWindow(float fPosX, float fPosY, int nXSize_LettersPerLine, char *sTitle, char *sBody)
{
#if !defined (DISABLE_DISPLAYTEXT)

	unsigned int dwActualWin;
	unsigned int dwFlags = eDisplayText_ADJUST_SIZE_ALWAYS;
	unsigned int dwBodyTextColor, dwBodyBackgroundColor;

	/* If no text is specified, return an error */
	if(!sBody && !sTitle) return 0xFFFFFFFF;

	/* If no title is specified, body text colours are different */
	if(!sTitle)
	{
		dwBodyTextColor			= RGBA(0xFF, 0xFF, 0x30, 0xFF);
		dwBodyBackgroundColor	= RGBA(0x20, 0x20, 0xB0, 0xE0);
	}
	else
	{
		dwBodyTextColor			= RGBA(0xFF, 0xFF, 0xFF, 0xFF);
		dwBodyBackgroundColor	= RGBA(0x20, 0x30, 0xFF, 0xE0);
	}

	/* Set window flags depending on title and body text were specified */
	if(!sBody)		dwFlags |= eDisplayText_DEACTIVATE_WIN;
	if(!sTitle)		dwFlags |= eDisplayText_DEACTIVATE_TITLE;

	/* Create window */
	dwActualWin = InitWindow(nXSize_LettersPerLine, (sTitle==NULL) ? 1:50);

	/* Set window properties */
	SetWindow(dwActualWin, dwBodyBackgroundColor, dwBodyTextColor, 0.5f, fPosX, fPosY, 20.0f, 20.0f);

	/* Set title */
	if (sTitle)
		SetTitle(dwActualWin, RGBA(0x20, 0x20, 0xB0, 0xE0), 0.6f, RGBA(0xFF, 0xFF, 0x30, 0xFF), sTitle, RGBA(0xFF, 0xFF, 0x30, 0xFF), (char *)"");

	/* Set window text */
	if (sBody)
		SetText(dwActualWin, sBody);

	/* Set window flags */
	SetWindowFlags(dwActualWin, dwFlags);

	m_pWin[dwActualWin].bNeedUpdated = true;

	/* Return window handle */
	return dwActualWin;

#else
	return 0;
#endif
}

//
// dwBufferSizeX		Buffer width
// dwBufferSizeY		Buffer height
// Window handle
// Allocate a buffer for a newly-created window and return its
// handle.
//
unsigned int CDisplayText::InitWindow(unsigned int dwBufferSizeX, unsigned int dwBufferSizeY)
{
#if !defined (DISABLE_DISPLAYTEXT)

	unsigned int		dwCurrentWin;

	/* Find the first available window */
	for (dwCurrentWin=1; dwCurrentWin<DISPLAYTEXT_MAX_WINDOWS; dwCurrentWin++)
	{
		/* If this window available? */
		if (!(m_pWin[dwCurrentWin].dwFlags & DisplayText_WIN_EXIST))
		{
			/* Window available, exit loop */
			break;
		}
	}

	/* No more windows available? */
	if (dwCurrentWin == DISPLAYTEXT_MAX_WINDOWS)
	{
		_RPT0(_CRT_WARN,"\nDisplayTextCreateWindow WARNING: DISPLAYTEXT_MAX_WINDOWS overflow.\n");
		return 0;
	}

	/* Set flags */
	m_pWin[dwCurrentWin].dwFlags = DisplayText_WIN_TITLE  | DisplayText_WIN_EXIST | DisplayText_WIN_ACTIVE;

	/* Text Buffer */
	m_pWin[dwCurrentWin].dwBufferSizeX = dwBufferSizeX + 1;
	m_pWin[dwCurrentWin].dwBufferSizeY = dwBufferSizeY;
	m_pWin[dwCurrentWin].pTextBuffer  = (char*) malloc((dwBufferSizeX+2)*(dwBufferSizeY+2) * sizeof(char));// (char *)calloc((dwBufferSizeX+2)*(dwBufferSizeY+2), sizeof(char));
	memset(m_pWin[dwCurrentWin].pTextBuffer, 0, sizeof((dwBufferSizeX+2)*(dwBufferSizeY+2) * sizeof(char)));
	m_pWin[dwCurrentWin].bTitleTextL  = (char*) malloc(MAX_LETTERS * sizeof(char)); // (char *)calloc(MAX_LETTERS, sizeof(char));
	memset(m_pWin[dwCurrentWin].bTitleTextL, 0, MAX_LETTERS * sizeof(char));
	m_pWin[dwCurrentWin].bTitleTextR  = (char*) malloc(MAX_LETTERS * sizeof(char)); //(char *)calloc(MAX_LETTERS, sizeof(char));
	memset(m_pWin[dwCurrentWin].bTitleTextR, 0, MAX_LETTERS * sizeof(char));

	/* Memory allocation failed */
	if (!m_pWin[dwCurrentWin].pTextBuffer || !m_pWin[dwCurrentWin].bTitleTextL || !m_pWin[dwCurrentWin].bTitleTextR)
	{
		_RPT0(_CRT_WARN,"\nDisplayTextCreateWindow : No memory enough for Text Buffer.\n");
		return 0;
	}

	/* Title */
	m_pWin[dwCurrentWin].fTitleFontSize	= 1.0f;
	m_pWin[dwCurrentWin].dwTitleFontColorL = RGBA(0xFF, 0xFF, 0x00, 0xFF);
	m_pWin[dwCurrentWin].dwTitleFontColorR = RGBA(0xFF, 0xFF, 0x00, 0xFF);
	m_pWin[dwCurrentWin].dwTitleBaseColor  = RGBA(0x30, 0x30, 0xFF, 0xFF); /* Dark Blue */

	/* Window */
	m_pWin[dwCurrentWin].fWinFontSize		= 0.5f;
	m_pWin[dwCurrentWin].dwWinFontColor	= RGBA(0xFF, 0xFF, 0xFF, 0xFF);
	m_pWin[dwCurrentWin].dwWinBaseColor	= RGBA(0x80, 0x80, 0xFF, 0xFF); /* Light Blue */
	m_pWin[dwCurrentWin].fWinPos[0]		= 0.0f;
	m_pWin[dwCurrentWin].fWinPos[1]		= 0.0f;
	m_pWin[dwCurrentWin].fWinSize[0]		= 20.0f;
	m_pWin[dwCurrentWin].fWinSize[1]		= 20.0f;
	m_pWin[dwCurrentWin].fZPos		        = 0.0f;
	m_pWin[dwCurrentWin].dwSort		    = 0;

	m_pWin[dwCurrentWin].bNeedUpdated = true;

	dwCurrentWin++;

	/* Returning the handle */
	return (dwCurrentWin-1);

#else
	return 0;
#endif
}

//
// dwWin		Window handle
// Delete the window referenced by dwWin.
//
void CDisplayText::DeleteWindow(unsigned int dwWin)
{
#if !defined (DISABLE_DISPLAYTEXT)

	int i;

	/* Release VertexBuffer */
	free(m_pWin[dwWin].pTitleVtxL);
	free(m_pWin[dwWin].pTitleVtxR);
	free(m_pWin[dwWin].pWindowVtxTitle);
	free(m_pWin[dwWin].pWindowVtxText);
	for(i=0; i<255; i++)
		free(m_pWin[dwWin].pLineVtx[i]);

	/* Only delete window if it exists */
	if(m_pWin[dwWin].dwFlags & DisplayText_WIN_EXIST)
	{
		free(m_pWin[dwWin].pTextBuffer);
		free(m_pWin[dwWin].bTitleTextL);
		free(m_pWin[dwWin].bTitleTextR);
	}

	/* Reset flags */
	m_pWin[dwWin].dwFlags = 0;

#endif
}

//
// Delete all windows.
//
void CDisplayText::DeleteAllWindows()
{
#if !defined (DISABLE_DISPLAYTEXT)

	int unsigned i;

	for (i=0; i<DISPLAYTEXT_MAX_WINDOWS; i++)
		DeleteWindow (i);

#endif
}

//
// dwWin
// This function MUST be called between a BeginScene/EndScene
// pair as it uses D3D render primitive calls.
// DisplayTextSetTextures(...) must have been called beforehand.
//
void CDisplayText::DisplayWindow(unsigned int dwWin)
{
#if !defined (DISABLE_DISPLAYTEXT)

	unsigned int	i;
	float			fTitleSize = 0.0f;

	/* No textures! so... no window */

	if (!m_bTexturesSet)
	{
		_RPT0(_CRT_WARN,"DisplayTextDisplayWindow : You must call DisplayTextSetTextures()\nbefore using this function!!!\n");
		return;
	}

	/* Update Vertex data only when needed */
	if(m_pWin[dwWin].bNeedUpdated)
	{
		/* TITLE */
		if(m_pWin[dwWin].dwFlags & DisplayText_WIN_TITLE)
		{
			/* Set title size */
			if(m_pWin[dwWin].fTitleFontSize < 0.0f)
				fTitleSize = 8.0f + 16.0f;
			else
				fTitleSize = m_pWin[dwWin].fTitleFontSize * 23.5f + 16.0f;

			/* Title */
			UpdateTitleVertexBuffer(dwWin);

			/* Background */
			if (!(m_pWin[dwWin].dwFlags & DisplayText_FULL_TRANS))
			{
				/* Draw title background */
				UpdateBackgroundWindow(
					dwWin, m_pWin[dwWin].dwTitleBaseColor,
					0.0f,
					m_pWin[dwWin].fWinPos[0],
					m_pWin[dwWin].fWinPos[1],
					m_pWin[dwWin].fWinSize[0],
					fTitleSize, &m_pWin[dwWin].pWindowVtxTitle);
			}
		}

		/* Main text */
		UpdateMainTextVertexBuffer(dwWin);

		UpdateBackgroundWindow(
			dwWin, m_pWin[dwWin].dwWinBaseColor,
			0.0f,
			m_pWin[dwWin].fWinPos[0],
			(m_pWin[dwWin].fWinPos[1] + fTitleSize),
			m_pWin[dwWin].fWinSize[0],
			m_pWin[dwWin].fWinSize[1], &m_pWin[dwWin].pWindowVtxText);

		/* Don't update until next change makes it needed */
		m_pWin[dwWin].bNeedUpdated = false;
	}

	// Ensure any previously drawn text has been submitted before drawing the window.
	Flush();

	/* Save current render states */
	APIRenderStates(0);

	/*
		DRAW TITLE
	*/
	if(m_pWin[dwWin].dwFlags & DisplayText_WIN_TITLE)
	{
		if (!(m_pWin[dwWin].dwFlags & DisplayText_FULL_TRANS))
		{
			DrawBackgroundWindowUP(dwWin, m_pWin[dwWin].pWindowVtxTitle, (m_pWin[dwWin].dwFlags & DisplayText_FULL_OPAQUE) ? true : false, (m_pWin[dwWin].dwFlags & DisplayText_NO_BORDER) ? false : true);
		}

		/* Left and Right text */
		DrawLineUP(m_pWin[dwWin].pTitleVtxL, m_pWin[dwWin].nTitleVerticesL);
		DrawLineUP(m_pWin[dwWin].pTitleVtxR, m_pWin[dwWin].nTitleVerticesR);
	}

	/*
		DRAW WINDOW
	*/
	if (m_pWin[dwWin].dwFlags & DisplayText_WIN_ACTIVE)
	{
		/* Background */
		if (!(m_pWin[dwWin].dwFlags & DisplayText_FULL_TRANS))
		{
			DrawBackgroundWindowUP(dwWin, m_pWin[dwWin].pWindowVtxText, (m_pWin[dwWin].dwFlags & DisplayText_FULL_OPAQUE) ? true : false, (m_pWin[dwWin].dwFlags & DisplayText_NO_BORDER) ? false : true);
		}

		/* Text, line by line */
		for (i=0; i<m_pWin[dwWin].dwBufferSizeY; i++)
		{
			DrawLineUP(m_pWin[dwWin].pLineVtx[i], m_pWin[dwWin].nLineVertices[i]);
		}
	}

	/* Restore render states */
	APIRenderStates(1);

#endif
}

// 
// dwWin		Window handle
// Format		Format string
// Feed the text buffer of window referenced by dwWin.
// This function accepts formatting in the printf way.
//
void CDisplayText::SetText(unsigned int dwWin, const char *Format, ...)
{
#if !defined (DISABLE_DISPLAYTEXT)

	va_list			args;
	unsigned int			i;
	unsigned int			dwBufferSize, dwTotalLength = 0;
	unsigned int			dwPosBx, dwPosBy, dwSpcPos;
	char			bChar;
	unsigned int			dwCursorPos;
	static char	sText[MAX_LETTERS+1];

	/* If window doesn't exist then return from function straight away */
	if (!(m_pWin[dwWin].dwFlags & DisplayText_WIN_EXIST))
		return;
	// Empty the window buffer
	memset(m_pWin[dwWin].pTextBuffer, 0, m_pWin[dwWin].dwBufferSizeX * m_pWin[dwWin].dwBufferSizeY * sizeof(char));

	/* Reading the arguments to create our Text string */
	va_start(args,Format);
	vsprintf(sText, Format, args);		// Could use _vsnprintf but e.g. LinuxVP does not support it
	va_end(args);

	dwCursorPos	= 0;

	m_pWin[dwWin].bNeedUpdated = true;

	/* Compute buffer size */
	dwBufferSize = (m_pWin[dwWin].dwBufferSizeX+1) * (m_pWin[dwWin].dwBufferSizeY+1);

	/* Compute length */
	while(dwTotalLength < dwBufferSize && sText[dwTotalLength] != 0)
		dwTotalLength++;

	/* X and Y pointer position */
	dwPosBx = 0;
	dwPosBy = 0;

	/* Process each character */
	for (i=0; i<dwTotalLength; i++)
	{
		/* Get current character in string */
		bChar = sText[i];

		/* Space (for word wrap only) */
		if (bChar == ' ')
		{
			/* Looking for the next space (or return or end) */
			dwSpcPos = 1;
			do 
			{
				bChar = sText[i + dwSpcPos++];
			}
			while (bChar==' ' || bChar==0x0A || bChar==0);
			bChar = ' ';

			/*
				Humm, if this word is longer than the buffer don't move it.
				Otherwise check if it is at the end and create a return.
			*/
			if (dwSpcPos<m_pWin[dwWin].dwBufferSizeX && (dwPosBx+dwSpcPos)>m_pWin[dwWin].dwBufferSizeX)
			{
				/* Set NULL character */
				m_pWin[dwWin].pTextBuffer[dwCursorPos++] = 0;

				dwPosBx = 0;
				dwPosBy++;

				/* Set new cursor position */
				dwCursorPos = dwPosBy * m_pWin[dwWin].dwBufferSizeX;

				/* Don't go any further */
				continue;
			}
		}

		/* End of line */
		if (dwPosBx == (m_pWin[dwWin].dwBufferSizeX-1))
		{
			m_pWin[dwWin].pTextBuffer[dwCursorPos++] = 0;
			dwPosBx = 0;
			dwPosBy++;
		}

		/* Vertical Scroll */
		if (dwPosBy >= m_pWin[dwWin].dwBufferSizeY)
		{
			memcpy(m_pWin[dwWin].pTextBuffer,
				m_pWin[dwWin].pTextBuffer + m_pWin[dwWin].dwBufferSizeX,
				(m_pWin[dwWin].dwBufferSizeX-1) * m_pWin[dwWin].dwBufferSizeY);

			dwCursorPos -= m_pWin[dwWin].dwBufferSizeX;

			dwPosBx = 0;
			dwPosBy--;
		}

		/* Return */
		if (bChar == 0x0A)
		{
			/* Set NULL character */
			m_pWin[dwWin].pTextBuffer[dwCursorPos++] = 0;

			dwPosBx = 0;
			dwPosBy++;

			dwCursorPos = dwPosBy * m_pWin[dwWin].dwBufferSizeX;

			/* Don't go any further */
			continue;
		}

		/* Storing our character */
		if (dwCursorPos<dwBufferSize)
		{
			m_pWin[dwWin].pTextBuffer[dwCursorPos++] = bChar;
		}

		/* Increase position */
		dwPosBx++;
	}

	/* Automatic adjust of the window size */
	if (m_pWin[dwWin].dwFlags & DisplayText_ADJUST_SIZE)
	{
		AdjustWindowSize(dwWin, 0);
	}

#endif
}

//
// dwWin			Window handle
// dwWinColor		Window colour
// dwFontColor		Font colour
// fFontSize		Font size
// fPosX			Window position X
// fPosY			Window position Y
// fSizeX			Window size X
// fSizeY			Window size Y
// Set attributes of window.
// Windows position and size are referred to a virtual screen
// of 100x100. (0,0) is the top-left corner and (100,100) the
// bottom-right corner.
// These values are the same for all resolutions.
//
void CDisplayText::SetWindow(unsigned int dwWin, unsigned int dwWinColor, unsigned int dwFontColor, float fFontSize,
						  float fPosX, float fPosY, float fSizeX, float fSizeY)
{
#if !defined (DISABLE_DISPLAYTEXT)

	/* Check if there is a real change */
	if(	m_pWin[dwWin].fWinFontSize		!= fFontSize ||
		m_pWin[dwWin].dwWinFontColor	!= dwFontColor ||
		m_pWin[dwWin].dwWinBaseColor	!= dwWinColor ||
		m_pWin[dwWin].fWinPos[0]		!= fPosX  * WindowWidth/100.0f ||
		m_pWin[dwWin].fWinPos[1]		!= fPosY  * WindowHeight/100.0f ||
		m_pWin[dwWin].fWinSize[0]		!= fSizeX * WindowWidth/100.0f ||
		m_pWin[dwWin].fWinSize[1]		!= fSizeY * WindowHeight/100.0f)
	{
		/* Set window properties */
		m_pWin[dwWin].fWinFontSize		= fFontSize;
		m_pWin[dwWin].dwWinFontColor	= dwFontColor;
		m_pWin[dwWin].dwWinBaseColor	= dwWinColor;
		m_pWin[dwWin].fWinPos[0]		= fPosX  * WindowWidth/100.0f;
		m_pWin[dwWin].fWinPos[1]		= fPosY  * WindowHeight/100.0f;
		m_pWin[dwWin].fWinSize[0]		= fSizeX * WindowWidth/100.0f;
		m_pWin[dwWin].fWinSize[1]		= fSizeY * WindowHeight/100.0f;

		m_pWin[dwWin].bNeedUpdated = true;
	}

#endif
}

//
// dwWin				Window handle
// dwBackgroundColor	Background color
// fFontSize			Font size
// dwFontColorLeft
// sTitleLeft
// dwFontColorRight
// sTitleRight
// Set window title.
//
void CDisplayText::SetTitle(unsigned int dwWin, unsigned int dwBackgroundColor, float fFontSize,
						 unsigned int dwFontColorLeft, char *sTitleLeft,
						 unsigned int dwFontColorRight, char *sTitleRight)
{
#if !defined (DISABLE_DISPLAYTEXT)

	free(m_pWin[dwWin].pTitleVtxL);
	free(m_pWin[dwWin].pTitleVtxR);

	if(sTitleLeft)  memcpy(m_pWin[dwWin].bTitleTextL, sTitleLeft , _MIN(MAX_LETTERS-1, strlen(sTitleLeft )+1));
	if(sTitleRight) memcpy(m_pWin[dwWin].bTitleTextR, sTitleRight, _MIN(MAX_LETTERS-1, strlen(sTitleRight)+1));

	/* Set title properties */
	m_pWin[dwWin].fTitleFontSize		= fFontSize;
	m_pWin[dwWin].dwTitleFontColorL	= dwFontColorLeft;
	m_pWin[dwWin].dwTitleFontColorR	= dwFontColorRight;
	m_pWin[dwWin].dwTitleBaseColor	= dwBackgroundColor;
	m_pWin[dwWin].fTextRMinPos		= GetLength(m_pWin[dwWin].fTitleFontSize, m_pWin[dwWin].bTitleTextL) + 10.0f;
	m_pWin[dwWin].bNeedUpdated		= true;

#endif
}

//
// dwWin				Window handle
// dwFlags				Flags
// Set flags for window referenced by dwWin.
// A list of flag can be found at the top of this header.
//
void CDisplayText::SetWindowFlags(unsigned int dwWin, unsigned int dwFlags)
{
#if !defined (DISABLE_DISPLAYTEXT)

	/* Check if there is need of updating vertex buffers */
	if(	dwFlags & eDisplayText_ACTIVATE_TITLE ||
		dwFlags & eDisplayText_DEACTIVATE_TITLE ||
		dwFlags & eDisplayText_ADJUST_SIZE_ALWAYS)
		m_pWin[dwWin].bNeedUpdated = true;

	/* Set window flags */
	if (dwFlags & eDisplayText_ACTIVATE_WIN)		m_pWin[dwWin].dwFlags |= DisplayText_WIN_ACTIVE;
	if (dwFlags & eDisplayText_DEACTIVATE_WIN)	m_pWin[dwWin].dwFlags &= ~DisplayText_WIN_ACTIVE;
	if (dwFlags & eDisplayText_ACTIVATE_TITLE)	m_pWin[dwWin].dwFlags |= DisplayText_WIN_TITLE;
	if (dwFlags & eDisplayText_DEACTIVATE_TITLE) m_pWin[dwWin].dwFlags &= ~DisplayText_WIN_TITLE;
	if (dwFlags & eDisplayText_FULL_OPAQUE)		m_pWin[dwWin].dwFlags |= DisplayText_FULL_OPAQUE;
	if (dwFlags & eDisplayText_FULL_TRANS)		m_pWin[dwWin].dwFlags |= DisplayText_FULL_TRANS;

	if (dwFlags & eDisplayText_ADJUST_SIZE_ALWAYS)
	{
		m_pWin[dwWin].dwFlags |= DisplayText_ADJUST_SIZE;
		AdjustWindowSize(dwWin, 0);
	}

	if (dwFlags & eDisplayText_NO_BORDER)	m_pWin[dwWin].dwFlags |= DisplayText_NO_BORDER;

#endif
}

//
// dwWin				Window handle
// dwMode				dwMode 0 = Both, dwMode 1 = X only,  dwMode 2 = Y only
// Calculates window size so that all text fits in the window.
//
void CDisplayText::AdjustWindowSize(unsigned int dwWin, unsigned int dwMode)
{
#if !defined (DISABLE_DISPLAYTEXT)

	int unsigned i;
	unsigned int dwPointer = 0;
	float fMax = 0.0f, fLength;

	if (dwMode==1 || dwMode==0)
	{
		/* Title horizontal Size */
		if(m_pWin[dwWin].dwFlags & DisplayText_WIN_TITLE)
		{
			fMax = GetLength(m_pWin[dwWin].fTitleFontSize, m_pWin[dwWin].bTitleTextL);

			if (m_pWin[dwWin].bTitleTextR)
			{
				fMax += GetLength(m_pWin[dwWin].fTitleFontSize, m_pWin[dwWin].bTitleTextR) + 12.0f;
			}
		}

		/* Body horizontal size (line by line) */
		for (i=0; i<m_pWin[dwWin].dwBufferSizeY; i++)
		{
			fLength = GetLength(m_pWin[dwWin].fWinFontSize, (m_pWin[dwWin].pTextBuffer + dwPointer));

			if (fLength > fMax) fMax = fLength;

			dwPointer += m_pWin[dwWin].dwBufferSizeX;
		}

		m_pWin[dwWin].fWinSize[0] = fMax - 2.0f + 16.0f;
	}

	/* Vertical Size */
	if(dwMode==0 || dwMode==2)
	{
		if(m_pWin[dwWin].dwBufferSizeY < 2)
		{
			i = 0;
		}
		else
		{
			/* Looking for the last line */
			i=m_pWin[dwWin].dwBufferSizeY;
			while(i)
			{
				--i;
				if (m_pWin[dwWin].pTextBuffer[m_pWin[dwWin].dwBufferSizeX * i])
					break;
			}
		}

		if (m_pWin[dwWin].fWinFontSize>0)
			m_pWin[dwWin].fWinSize[1] = (float)(i+1) * LINES_SPACING * m_pWin[dwWin].fWinFontSize + 16.0f;
		else
			m_pWin[dwWin].fWinSize[1] = ((float)(i+1) * 12.0f) + 16.0f;
	}

	m_pWin[dwWin].bNeedUpdated = true;

#endif
}

//
// pfWidth				Width of the string in pixels
// pfHeight			Height of the string in pixels
// fFontSize			Font size
// sString				String to take the size of
// Returns the size of a string in pixels.
//
void CDisplayText::GetSize(
	float		* const pfWidth,
	float		* const pfHeight,
	const float	fFontSize,
	const char	* sString)
{
#if !defined (DISABLE_DISPLAYTEXT)

	unsigned char Val;
	float fScale, fSize;

	if(sString == NULL) {
		if(pfWidth)
			*pfWidth = 0;
		if(pfHeight)
			*pfHeight = 0;
		return;
	}

	if(fFontSize > 0.0f) /* Arial font */
	{
		fScale = fFontSize;
		fSize  = 0.0f;

		Val = *sString++;
		while(Val)
		{
			if(Val==' ')
				Val = '0';

			if(Val>='0' && Val <= '9')
				Val = '0'; /* That's for fixing the number width */

			fSize += DisplayTextSize_Bold[Val] * 40.0f * fScale ;

			/* these letters are too narrow due to a bug in the table */
			if(Val=='i' || Val == 'l' || Val == 'j')
				fSize += 0.4f* fScale;
			Val = *sString++;
		}

		if(pfHeight)
			*pfHeight = m_fScreenScale[1] * fScale * 27.0f * (100.0f / WindowWidth);//CHANGED THIS>>>>480.0f);
	}
	else /* System font */
	{
		fScale = 255.0f;
		fSize  = 0.0f;

		Val = *sString++;
		while (Val)
		{
			if(Val == ' ') {
				fSize += 5.0f;
				continue;
			}

			if(Val>='0' && Val <= '9')
				Val = '0'; /* That's for fixing the number width */

			fSize += DisplayTextSize_System[Val]  * fScale * (100.0f / WindowWidth);//CHANGED THIS>>>>>480.0f);
			Val = *sString++;
		}

		if(pfHeight)
			*pfHeight = m_fScreenScale[1] * 12.0f;
	}

	if(pfWidth)
		*pfWidth = fSize;

#endif
}

//
// dwScreenX		Screen resolution X
// dwScreenY		Screen resolution Y
// Returns the current resolution used by DisplayText
//
void CDisplayText::GetAspectRatio(unsigned int *dwScreenX, unsigned int *dwScreenY)
{
#if !defined (DISABLE_DISPLAYTEXT)

	*dwScreenX = (int)(WindowWidth * m_fScreenScale[0]);
	*dwScreenY = (int)(WindowHeight * m_fScreenScale[1]);

#endif
}

/*************************************************************
*					 PRIVATE FUNCTIONS						 *
**************************************************************/

// 
// true if succesful, false otherwise.
// Draw a generic rectangle (with or without border).
//
bool CDisplayText::UpdateBackgroundWindow(unsigned int dwWin, unsigned int Color, float fZPos, float fPosX, float fPosY, float fSizeX, float fSizeY, SDisplayTextAPIVertex **ppVtx)
{
	int				i;
	SDisplayTextAPIVertex	*vBox;
	float			fU[] = { 0.0f, 0.0f, 6.0f, 6.0f, 10.0f,10.0f, 16.0f,16.0f,10.0f,16.0f,10.0f,16.0f,6.0f,6.0f,0.0f,0.0f};
	float			fV[] = { 0.0f, 6.0f, 0.0f, 6.0f, 0.0f, 6.0f, 0.0f, 6.0f, 10.0f, 10.0f, 16.0f,16.0f, 16.0f, 10.0f, 16.0f, 10.0f};

	/* Create our vertex buffers */
	if(*ppVtx==0)
	{
		*ppVtx = (SDisplayTextAPIVertex*) malloc(16*sizeof(SDisplayTextAPIVertex));
		memset(*ppVtx, 0, 16*sizeof(SDisplayTextAPIVertex));
	}
	vBox = *ppVtx;


	/* Removing the border */
	fSizeX -= 16.0f ;
	fSizeY -= 16.0f ;

	/* Set Z position, color and texture coordinates in array */
	for (i=0; i<16; i++)
	{
		vBox[i].sz		= f2vt(fZPos);
		//vBox[i].color	= Color;
		vBox[i].r = ((float)((Color & 0xFF)>>0)) / 255.0f;
		vBox[i].g = ((float)((Color & 0xFF00)>>8)) / 255.0f;
		vBox[i].b = ((float)((Color & 0xFF0000)>>16)) / 255.0f;
		vBox[i].a = ((float)((Color & 0xFF000000)>>24)) / 255.0f;

		vBox[i].tu		= f2vt(fU[i]/16.0f);
		vBox[i].tv		= f2vt(1.0f - fV[i]/16.0f);
	}

	/* Set coordinates in array */
	vBox[0].sx = f2vt((fPosX + fU[0]) * m_fScreenScale[0]);
	vBox[0].sy = f2vt((fPosY + fV[0]) * m_fScreenScale[1]);

	vBox[1].sx = f2vt((fPosX + fU[1]) * m_fScreenScale[0]);
	vBox[1].sy = f2vt((fPosY + fV[1]) * m_fScreenScale[1]);

	vBox[2].sx = f2vt((fPosX + fU[2]) * m_fScreenScale[0]);
	vBox[2].sy = f2vt((fPosY + fV[2]) * m_fScreenScale[1]);

	vBox[3].sx = f2vt((fPosX + fU[3]) * m_fScreenScale[0]);
	vBox[3].sy = f2vt((fPosY + fV[3]) * m_fScreenScale[1]);

	vBox[4].sx = f2vt((fPosX + fU[4] + fSizeX) * m_fScreenScale[0]);
	vBox[4].sy = f2vt((fPosY + fV[4]) * m_fScreenScale[1]);

	vBox[5].sx = f2vt((fPosX + fU[5] + fSizeX) * m_fScreenScale[0]);
	vBox[5].sy = f2vt((fPosY + fV[5]) * m_fScreenScale[1]);

	vBox[6].sx = f2vt((fPosX + fU[6] + fSizeX) * m_fScreenScale[0]);
	vBox[6].sy = f2vt((fPosY + fV[6]) * m_fScreenScale[1]);

	vBox[7].sx = f2vt((fPosX + fU[7] + fSizeX) * m_fScreenScale[0]);
	vBox[7].sy = f2vt((fPosY + fV[7]) * m_fScreenScale[1]);

	vBox[8].sx = f2vt((fPosX + fU[8] + fSizeX) * m_fScreenScale[0]);
	vBox[8].sy = f2vt((fPosY + fV[8] + fSizeY) * m_fScreenScale[1]);

	vBox[9].sx = f2vt((fPosX + fU[9] + fSizeX) * m_fScreenScale[0]);
	vBox[9].sy = f2vt((fPosY + fV[9] + fSizeY) * m_fScreenScale[1]);

	vBox[10].sx = f2vt((fPosX + fU[10] + fSizeX) * m_fScreenScale[0]);
	vBox[10].sy = f2vt((fPosY + fV[10] + fSizeY) * m_fScreenScale[1]);

	vBox[11].sx = f2vt((fPosX + fU[11] + fSizeX) * m_fScreenScale[0]);
	vBox[11].sy = f2vt((fPosY + fV[11] + fSizeY) * m_fScreenScale[1]);

	vBox[12].sx = f2vt((fPosX + fU[12]) * m_fScreenScale[0]);
	vBox[12].sy = f2vt((fPosY + fV[12] + fSizeY) * m_fScreenScale[1]);

	vBox[13].sx = f2vt((fPosX + fU[13]) * m_fScreenScale[0]);
	vBox[13].sy = f2vt((fPosY + fV[13] + fSizeY) * m_fScreenScale[1]);

	vBox[14].sx = f2vt((fPosX + fU[14]) * m_fScreenScale[0]);
	vBox[14].sy = f2vt((fPosY + fV[14] + fSizeY) * m_fScreenScale[1]);

	vBox[15].sx = f2vt((fPosX + fU[15]) * m_fScreenScale[0]);
	vBox[15].sy = f2vt((fPosY + fV[15] + fSizeY) * m_fScreenScale[1]);

	if(bScreenRotate)
	{
		Rotate(vBox, 16);
	}

	/* No problem occured */
	return true;
}

unsigned int CDisplayText::UpdateLine(const unsigned int dwWin, const float fZPos, float XPos, float YPos, const float fScale, const unsigned int Colour, const char * const Text, SDisplayTextAPIVertex * const pVertices)
{
	unsigned	i=0, VertexCount=0;
	unsigned	Val;
	float		XSize = 0.0f, XFixBug,	YSize = 0, TempSize;
	float		UPos,	VPos;
	float		USize,	VSize;
	float		fWinClipX[2],fWinClipY[2];
	float		fScaleX, fScaleY, fPreXPos;

	/* Nothing to update */
	if (Text==NULL) return 0;

	_ASSERT(m_pWin[dwWin].dwFlags & DisplayText_WIN_EXIST || !dwWin);

	if (fScale>0)
	{
		fScaleX = m_fScreenScale[0] * fScale * 255.0f;
		fScaleY = m_fScreenScale[1] * fScale * 27.0f;
	}
	else
	{
		fScaleX = m_fScreenScale[0] * 255.0f;
		fScaleY = m_fScreenScale[1] * 12.0f;
	}

	XPos *= m_fScreenScale[0];
	YPos *= m_fScreenScale[1];
#if __IPHONE_OS_VERSION_MAX_ALLOWED >= 30200
	if(GetDeviceType()==IPAD_DEVICE)
		YPos /=2.0f;
#endif

	fPreXPos = XPos;

	/*
		Calculating our margins
	*/
	if (dwWin)
	{
		fWinClipX[0] = (m_pWin[dwWin].fWinPos[0] + 6.0f) * m_fScreenScale[0];
		fWinClipX[1] = (m_pWin[dwWin].fWinPos[0] + m_pWin[dwWin].fWinSize[0] - 6.0f) * m_fScreenScale[0];

		fWinClipY[0] = (m_pWin[dwWin].fWinPos[1] + 6.0f) * m_fScreenScale[1];
		fWinClipY[1] = (m_pWin[dwWin].fWinPos[1] + m_pWin[dwWin].fWinSize[1]  + 9.0f) * m_fScreenScale[1];

		if(m_pWin[dwWin].dwFlags & DisplayText_WIN_TITLE)
		{
			if (m_pWin[dwWin].fTitleFontSize>0)
			{
				fWinClipY[0] +=  m_pWin[dwWin].fTitleFontSize * 25.0f  * m_fScreenScale[1];
				fWinClipY[1] +=  m_pWin[dwWin].fTitleFontSize * 25.0f *  m_fScreenScale[1];
			}
			else
			{
				fWinClipY[0] +=  10.0f * m_fScreenScale[1];
				fWinClipY[1] +=  8.0f  * m_fScreenScale[1];
			}
		}
	}

	while (true)
	{
		Val = (int)Text[i++];

		/* End of the string */
		if (Val==0 || i>MAX_LETTERS) break;

		/* It is SPACE so don't draw and carry on... */
		if (Val==' ')
		{
			if (fScale>0)	XPos += 10.0f/255.0f * fScaleX;
			else			XPos += 5.0f * m_fScreenScale[0];
			continue;
		}

		/* It is SPACE so don't draw and carry on... */
		if (Val=='#')
		{
			if (fScale>0)	XPos += 1.0f/255.0f * fScaleX;
			else			XPos += 5.0f * m_fScreenScale[0];
			continue;
		}

		/* It is RETURN so jump a line */
		if (Val==0x0A)
		{
			XPos = fPreXPos - XSize;
			YPos += YSize;
			continue;
		}

		/* If fScale is negative then select the small set of letters (System) */
		if (fScale < 0.0f)
		{
			XPos    += XSize;
			UPos    =  DisplayTextU_System[Val];
			VPos    =  DisplayTextV_System[Val] - 0.0001f; /* Some cards need this little bit */
			YSize   =  fScaleY;
			XSize   =  DisplayTextSize_System[Val] * fScaleX;
			USize	=  DisplayTextSize_System[Val];
			VSize	=  12.0f/255.0f;
		}
		else /* Big set of letters (Bold) */
		{
			XPos    += XSize;
			UPos    =  DisplayTextU_Bold[Val];
			VPos    =  DisplayTextV_Bold[Val] - 1.0f/230.0f;
			YSize   =  fScaleY;
			XSize   =  DisplayTextSize_Bold[Val] * fScaleX;
			USize	=  DisplayTextSize_Bold[Val];
			VSize	=  29.0f/255.0f;
		}

		/*
			CLIPPING
		*/
		XFixBug = XSize;

		if (0)//dwWin) /* for dwWin==0 (screen) no clipping */
		{
			/* Outside */
			if (XPos>fWinClipX[1]  ||  (YPos)>fWinClipY[1])
			{
				continue;
			}

			/* Clip X */
			if (XPos<fWinClipX[1] && XPos+XSize > fWinClipX[1])
			{
				TempSize = XSize;

				XSize = fWinClipX[1] - XPos;

				if (fScale < 0.0f)
					USize	=  DisplayTextSize_System[Val] * (XSize/TempSize);
				else
					USize	=  DisplayTextSize_Bold[Val] * (XSize/TempSize);
			}

			/*
				Clip Y
			*/
			if (YPos<fWinClipY[1] && YPos+YSize > fWinClipY[1])
			{
				TempSize = YSize;
				YSize = fWinClipY[1] - YPos;

				if(fScale < 0.0f)
				 	VSize	=  (YSize/TempSize)*12.0f/255.0f;
				else
					VSize	=  (YSize/TempSize)*28.0f/255.0f;
			}
		}


		/* Filling vertex data */
		pVertices[VertexCount+0].sx		= f2vt(XPos);
		pVertices[VertexCount+0].sy		= f2vt(YPos);
		pVertices[VertexCount+0].sz		= f2vt(fZPos);
		pVertices[VertexCount+0].tu		= f2vt(UPos);
		pVertices[VertexCount+0].tv		= f2vt(VPos);

		pVertices[VertexCount+1].sx		= f2vt(XPos+XSize);
		pVertices[VertexCount+1].sy		= f2vt(YPos);
		pVertices[VertexCount+1].sz		= f2vt(fZPos);
		pVertices[VertexCount+1].tu		= f2vt(UPos+USize);
		pVertices[VertexCount+1].tv		= f2vt(VPos);

		pVertices[VertexCount+2].sx		= f2vt(XPos);
		pVertices[VertexCount+2].sy		= f2vt(YPos+YSize);
		pVertices[VertexCount+2].sz		= f2vt(fZPos);
		pVertices[VertexCount+2].tu		= f2vt(UPos);
		pVertices[VertexCount+2].tv		= f2vt(VPos-VSize);

		pVertices[VertexCount+3].sx		= f2vt(XPos+XSize);
		pVertices[VertexCount+3].sy		= f2vt(YPos+YSize);
		pVertices[VertexCount+3].sz		= f2vt(fZPos);
		pVertices[VertexCount+3].tu		= f2vt(UPos+USize);
		pVertices[VertexCount+3].tv		= f2vt(VPos-VSize);

		float r = ((float)((Colour & 0xFF)>>0)) / 255.0f;
		float g = ((float)((Colour & 0xFF00)>>8)) / 255.0f;
		float b = ((float)((Colour & 0xFF0000)>>16)) / 255.0f;
		float a = ((float)((Colour & 0xFF000000)>>24)) / 255.0f;
		
		pVertices[VertexCount+0].r = r;
		pVertices[VertexCount+0].g = g;
		pVertices[VertexCount+0].b = b;
		pVertices[VertexCount+0].a = a;
		
		pVertices[VertexCount+1].r = r;
		pVertices[VertexCount+1].g = g;
		pVertices[VertexCount+1].b = b;
		pVertices[VertexCount+1].a = a;
		
		pVertices[VertexCount+2].r = r;
		pVertices[VertexCount+2].g = g;
		pVertices[VertexCount+2].b = b;
		pVertices[VertexCount+2].a = a;
		
		pVertices[VertexCount+3].r = r;
		pVertices[VertexCount+3].g = g;
		pVertices[VertexCount+3].b = b;
		pVertices[VertexCount+3].a = a;
		
		VertexCount += 4;

		XSize = XFixBug;

		/* Fix number width */
		if (Val >='0' && Val <='9')
		{
			if (fScale < 0.0f)
				XSize = DisplayTextSize_System[(int)'0'] * fScaleX;
			else
				XSize = DisplayTextSize_Bold[(int)'0'] * fScaleX;
		}
	}

	if(bScreenRotate)
	{
		Rotate(pVertices, VertexCount);
	}

	return VertexCount;
}

void CDisplayText::DrawLineUP(SDisplayTextAPIVertex *pVtx, unsigned int nVertices)
{
	if(!nVertices)
		return;

	_ASSERT((nVertices % 4) == 0);
	_ASSERT((nVertices/4) < MAX_LETTERS);

	while(m_nVtxCache + (int)nVertices > m_nVtxCacheMax) 
	{
		if(m_nVtxCache + nVertices > MAX_CACHED_VTX) 
		{
			_RPT1(_CRT_WARN, "DisplayText: Out of space to cache text! (More than %d vertices!)\n", MAX_CACHED_VTX);
			return;
		}

		m_nVtxCacheMax	= _MIN(m_nVtxCacheMax * 2, MAX_CACHED_VTX);
		m_pVtxCache		= (SDisplayTextAPIVertex*) reallocEM(m_pVtxCache, sizeof(*m_pVtxCache), m_nVtxCacheMax * sizeof(*m_pVtxCache));
		_ASSERT(m_pVtxCache);
		_RPT1(_CRT_WARN, "DisplayText: TextCache increased to %d vertices.\n", m_nVtxCacheMax);
	}

	memcpy(&m_pVtxCache[m_nVtxCache], pVtx, nVertices * sizeof(*pVtx));
	m_nVtxCache += nVertices;
}


void CDisplayText::UpdateTitleVertexBuffer(unsigned int dwWin)
{
	float fRPos;
	unsigned int dwLenL = 0, dwLenR = 0;

	/* Doesn't exist */
	if (!(m_pWin[dwWin].dwFlags & DisplayText_WIN_EXIST) && dwWin)
		return;

	/* Allocate our buffers if needed */
	if(m_pWin[dwWin].pTitleVtxL==0 || m_pWin[dwWin].pTitleVtxR==0)
	{
		dwLenL = (unsigned int)strlen(m_pWin[dwWin].bTitleTextL);
		free(m_pWin[dwWin].pTitleVtxL);
		if(dwLenL)
		{
			m_pWin[dwWin].pTitleVtxL = (SDisplayTextAPIVertex*) malloc(dwLenL*4*sizeof(SDisplayTextAPIVertex));
			memset(m_pWin[dwWin].pTitleVtxL, 0, dwLenL*4*sizeof(SDisplayTextAPIVertex));
		}

		dwLenR = m_pWin[dwWin].bTitleTextR ? (unsigned int)strlen(m_pWin[dwWin].bTitleTextR) : 0;
		free(m_pWin[dwWin].pTitleVtxR);
		if(dwLenR)
		{
			m_pWin[dwWin].pTitleVtxR = (SDisplayTextAPIVertex*) malloc(dwLenR*4*sizeof(SDisplayTextAPIVertex));
			memset(m_pWin[dwWin].pTitleVtxR, 0, dwLenR*4*sizeof(SDisplayTextAPIVertex));
		}

	}

	/* Left title */
	if (dwLenL)
	{
		m_pWin[dwWin].nTitleVerticesL = UpdateLine(dwWin, 0.0f,
			(m_pWin[dwWin].fWinPos[0] + 6.0f),
			(m_pWin[dwWin].fWinPos[1] + 7.0f),
			m_pWin[dwWin].fTitleFontSize,
			m_pWin[dwWin].dwTitleFontColorL,
			m_pWin[dwWin].bTitleTextL,
			m_pWin[dwWin].pTitleVtxL);
	}
	else
	{
		m_pWin[dwWin].nTitleVerticesL = 0;
		m_pWin[dwWin].pTitleVtxL = NULL;
	}

	/* Right title */
	if (dwLenR)
	{
		/* Compute position */
		fRPos = GetLength(m_pWin[dwWin].fTitleFontSize,m_pWin[dwWin].bTitleTextR);

		fRPos = m_pWin[dwWin].fWinSize[0]  - fRPos - 6.0f;

		/* Check that we're not under minimum position */
		if(fRPos<m_pWin[dwWin].fTextRMinPos)
			fRPos = m_pWin[dwWin].fTextRMinPos;

		/* Add window position */
		fRPos += m_pWin[dwWin].fWinPos[0];

		/* Print text */
		m_pWin[dwWin].nTitleVerticesR = UpdateLine(dwWin, 0.0f,
			fRPos,
			m_pWin[dwWin].fWinPos[1] + 7.0f,
			m_pWin[dwWin].fTitleFontSize,
			m_pWin[dwWin].dwTitleFontColorR,
			m_pWin[dwWin].bTitleTextR,
			m_pWin[dwWin].pTitleVtxR);
	}
	else
	{
		m_pWin[dwWin].nTitleVerticesR = 0;
		m_pWin[dwWin].pTitleVtxR = NULL;
	}
}

void CDisplayText::UpdateMainTextVertexBuffer(unsigned int dwWin)
{
	int i;
	float		fNewPos, fTitleSize;
	unsigned int		dwPointer = 0, dwLen;

	/* Doesn't exist */
	if (!(m_pWin[dwWin].dwFlags & DisplayText_WIN_EXIST) && dwWin) return;

	/* No text to update vertices */
	if(m_pWin[dwWin].pTextBuffer==NULL) return;

	/* Well, once we've got our text, allocate it to draw it later */
	/* Text, line by line */
	for (i=0; i<(int)m_pWin[dwWin].dwBufferSizeY; i++)
	{
		/* line length */
		dwLen = (unsigned int)strlen(&m_pWin[dwWin].pTextBuffer[dwPointer]);
		if(dwLen==0)
		{
			m_pWin[dwWin].nLineVertices[i] = 0;
			m_pWin[dwWin].pLineVtx[i] = NULL;
		}
		else
		{
			/* Create Vertex Buffer (one per line) */
			if (m_pWin[dwWin].pLineVtx[i]==0)
			{
				m_pWin[dwWin].pLineVtx[i] = (SDisplayTextAPIVertex*) malloc(m_pWin[dwWin].dwBufferSizeX *4*sizeof(SDisplayTextAPIVertex));
				memset(m_pWin[dwWin].pLineVtx[i], 0, m_pWin[dwWin].dwBufferSizeX *4*sizeof(SDisplayTextAPIVertex));
			}

			/* Compute new text position */
			fTitleSize = 0.0f;
			if(m_pWin[dwWin].fTitleFontSize < 0.0f)
			{
				/* New position for alternate font */
				if(m_pWin[dwWin].dwFlags & DisplayText_WIN_TITLE)
					fTitleSize = 8.0f +16;
				fNewPos = fTitleSize + (float)(i * 12.0f);
			}
			else
			{
				/* New position for normal font */
				if(m_pWin[dwWin].dwFlags & DisplayText_WIN_TITLE)
					fTitleSize = m_pWin[dwWin].fTitleFontSize * 23.5f + 16.0f;
				fNewPos = fTitleSize + (float)(i * m_pWin[dwWin].fWinFontSize) * LINES_SPACING;
			}

			/* Print window text */
			m_pWin[dwWin].nLineVertices[i] = UpdateLine(dwWin, 0.0f,
				(m_pWin[dwWin].fWinPos[0] + 6.0f),
				(m_pWin[dwWin].fWinPos[1] + 6.0f + fNewPos),
				m_pWin[dwWin].fWinFontSize, m_pWin[dwWin].dwWinFontColor,
				&m_pWin[dwWin].pTextBuffer[dwPointer],
				m_pWin[dwWin].pLineVtx[i]);
		}

		/* Increase pointer */
		dwPointer += m_pWin[dwWin].dwBufferSizeX;
	}
}

float CDisplayText::GetLength(float fFontSize, char *sString)
{
	unsigned char Val;
	float fScale, fSize;

	if(sString==NULL) return 0.0f;

	if (fFontSize>=0) /* Arial font */
	{
		fScale = fFontSize * 255.0f;
		fSize  = 0.0f;

		Val = *sString++;
		while (Val)
		{
			if(Val==' ')
			{
				fSize += 10.0f * fFontSize;
			}
			else
			{
				if(Val>='0' && Val <= '9') Val = '0'; /* That's for fixing the number width */
				fSize += DisplayTextSize_Bold[Val] * fScale ;
			}
			Val = *sString++;
		}
	}
	else /* System font */
	{
		fScale = 255.0f;
		fSize  = 0.0f;

		Val = *sString++;
		while (Val)
		{
			if (Val==' ')
			{
				fSize += 5.0f;
			}
			else
			{
				if(Val>='0' && Val <= '9') Val = '0'; /* That's for fixing the number width */
				fSize += DisplayTextSize_System[Val]  * fScale;
			}
			Val = *sString++;
		}
	}

	return (fSize);
}

void CDisplayText::Rotate(SDisplayTextAPIVertex * const pv, const unsigned int nCnt)
{
	unsigned int	i;
	VERTTYPE		x, y;
	

	for(i = 0; i < nCnt; ++i)
	{
		x = VERTTYPEDIV((VERTTYPE&)pv[i].sx, f2vt(WindowWidth * m_fScreenScale[0]));
		y = VERTTYPEDIV((VERTTYPE&)pv[i].sy, f2vt(WindowHeight * m_fScreenScale[1]));
		
		(VERTTYPE&)pv[i].sx = VERTTYPEMUL(-y+f2vt(1.0f), f2vt(WindowWidth * m_fScreenScale[0]));
		(VERTTYPE&)pv[i].sy = VERTTYPEMUL(x, f2vt(WindowHeight * m_fScreenScale[1]));
		//(VERTTYPE&)pv[i].sx = VERTTYPEMUL(y, f2vt(WindowWidth * m_fScreenScale[0]));
		//(VERTTYPE&)pv[i].sy = VERTTYPEMUL(f2vt(1.0f) - x, f2vt(WindowHeight * m_fScreenScale[1]));
	}
}
