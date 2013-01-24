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
#ifndef DISPLAYTEXT_H_
#define DISPLAYTEXT_H_

#include <TargetConditionals.h>
#include <Availability.h>
#include "MemoryManager.h"

#define DISPLAYTEXT_MAX_WINDOWS				(512)
#define DISPLAYTEXT_MAX_RENDERABLE_LETTERS	(0xFFFF >> 2)

typedef enum {
	eDisplayText_ACTIVATE_WIN		=	0x01,
	eDisplayText_DEACTIVATE_WIN		=	0x02,
	eDisplayText_ACTIVATE_TITLE		=	0x04,
	eDisplayText_DEACTIVATE_TITLE	=	0x08,
	eDisplayText_FULL_OPAQUE		=	0x10,
	eDisplayText_FULL_TRANS			=	0x20,
	eDisplayText_ADJUST_SIZE_ALWAYS	=	0x40,
	eDisplayText_NO_BORDER			=	0x80
} EDisplayTextFlags;

typedef enum {
	eDisplayTextLogoNone  = 0x00,
	eDisplayTextLogoPVR = 0x02,
	eDisplayTextLogoIMG = 0x04
} EDisplayTextLogo;

// A structure for our vertex type
struct SDisplayTextAPIVertex
{
	VERTTYPE		sx, sy, sz, rhw;
	VERTTYPE		r, g, b, a;
	VERTTYPE		tu, tv;
};

// Internal implementation data
struct SDisplayTextWIN
{
	unsigned int			dwFlags;

	bool					bNeedUpdated;

	/* Text Buffer */
	char					*pTextBuffer;
	unsigned int			dwBufferSizeX;
	unsigned int			dwBufferSizeY;

	/* Title */
	float					fTitleFontSize;
	float					fTextRMinPos;
	unsigned int			dwTitleFontColorL;
	unsigned int			dwTitleFontColorR;
	unsigned int			dwTitleBaseColor;
	char					*bTitleTextL;
	char					*bTitleTextR;
	unsigned int			nTitleVerticesL;
	unsigned int			nTitleVerticesR;
	SDisplayTextAPIVertex	*pTitleVtxL;
	SDisplayTextAPIVertex	*pTitleVtxR;

	/* Window */
	float					fWinFontSize;
	unsigned int			dwWinFontColor;
	unsigned int			dwWinBaseColor;
	float					fWinPos[2];
	float					fWinSize[2];
	float					fZPos;
	unsigned int			dwSort;
	unsigned int			nLineVertices[255]; // every line of text is allocated and drawn apart.
	SDisplayTextAPIVertex	*pLineVtx[256];
	SDisplayTextAPIVertex	*pWindowVtxTitle;
	SDisplayTextAPIVertex	*pWindowVtxText;
};

struct SDisplayTextAPI;

class CDisplayText
{
public:
	CDisplayText();
	~CDisplayText();

	bool bScreenRotate;
//
// pContext		Context
// dwScreenX		Screen resolution along X
// dwScreenY		Screen resolution along Y
// true or false
// Initialization and texture upload. Should be called only once
// for a given context.
//
bool SetTextures(
	const unsigned int	dwScreenX,
	const unsigned int	dwScreenY,
	bool			bRotate = true); 

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
void DisplayText(float fPosX, float fPosY, const float fScale, unsigned int Colour, const char * const pszFormat, ...);

// 
// pszTitle			Title to display
// pszDescription		Description to display
// uDisplayLogo		1 = Display the logo
// Creates a default title with predefined position and colours.
// It displays as well company logos when requested:
// 0 = No logo
// 1 = PowerVR logo
// 2 = iGDK logo
//
void DisplayDefaultTitle(const char * const pszTitle, const char * const pszDescription, const unsigned int uDisplayLogo);

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
unsigned int CreateDefaultWindow(float fPosX, float fPosY, int nXSize_LettersPerLine, char *sTitle, char *sBody);

//
// dwBufferSizeX		Buffer width
// dwBufferSizeY		Buffer height
// Window handle
// Allocate a buffer for a newly-created window and return its
// handle.
//
	unsigned int InitWindow(unsigned int dwBufferSizeX, unsigned int dwBufferSizeY);

//
// dwWin		Window handle
// Delete the window referenced by dwWin.
//
void DeleteWindow(unsigned int dwWin);

//
// Delete all windows.
//
void DeleteAllWindows();

//
// dwWin
// This function MUST be called between a BeginScene/EndScene
// pair as it uses D3D render primitive calls.
// DisplayTextSetTextures(...) must have been called beforehand.
//
void DisplayWindow(unsigned int dwWin);

// 
// dwWin		Window handle
// Format		Format string
// Feed the text buffer of window referenced by dwWin.
// This function accepts formatting in the printf way.
//
void SetText(unsigned int dwWin, const char *Format, ...);

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
void SetWindow(unsigned int dwWin, unsigned int dwWinColor, unsigned int dwFontColor, float fFontSize,
							  float fPosX, float fPosY, float fSizeX, float fSizeY);
							  
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
void SetTitle(unsigned int dwWin, unsigned int dwBackgroundColor, float fFontSize,
							 unsigned int dwFontColorLeft, char *sTitleLeft,
							 unsigned int dwFontColorRight, char *sTitleRight);

//
// dwWin				Window handle
// dwFlags				Flags
// Set flags for window referenced by dwWin.
// A list of flag can be found at the top of this header.
//
void SetWindowFlags(unsigned int dwWin, unsigned int dwFlags);

//
// dwWin				Window handle
// dwMode				dwMode 0 = Both, dwMode 1 = X only,  dwMode 2 = Y only
// Calculates window size so that all text fits in the window.
//
void AdjustWindowSize(unsigned int dwWin, unsigned int dwMode);

//
// pfWidth				Width of the string in pixels
// pfHeight			Height of the string in pixels
// fFontSize			Font size
// sString				String to take the size of
// Returns the size of a string in pixels.
//
void GetSize(
		float		* const pfWidth,
		float		* const pfHeight,
		const float	fFontSize,
		const char	* sString);

//
// dwScreenX		Screen resolution X
// dwScreenY		Screen resolution Y
// Returns the current resolution used by DisplayText
//
void GetAspectRatio(unsigned int *dwScreenX, unsigned int *dwScreenY);

private:
// 
// true if succesful, false otherwise.
// Draw a generic rectangle (with or without border).
//
bool UpdateBackgroundWindow(unsigned int dwWin, unsigned int Color, float fZPos, float fPosX, float fPosY, float fSizeX, float fSizeY, SDisplayTextAPIVertex **ppVtx);


unsigned int UpdateLine(const unsigned int dwWin, const float fZPos, float XPos, float YPos, const float fScale, const unsigned int Colour, const char * const Text, SDisplayTextAPIVertex * const pVertices);


void DrawLineUP(SDisplayTextAPIVertex *pVtx, unsigned int nVertices);


void UpdateTitleVertexBuffer(unsigned int dwWin);

void UpdateMainTextVertexBuffer(unsigned int dwWin);


float GetLength(float fFontSize, char *sString);

void Rotate(SDisplayTextAPIVertex * const pv, const unsigned int nCnt);

private:
	SDisplayTextAPI			*m_pAPI;
	unsigned int			m_uLogoToDisplay;
	unsigned short			*m_pwFacesFont;
	SDisplayTextAPIVertex	*m_pPrint3dVtx;
	float					m_fScreenScale[2];
	bool					m_bTexturesSet;
	SDisplayTextAPIVertex	*m_pVtxCache;
	int						m_nVtxCache;
	int						m_nVtxCacheMax;
	SDisplayTextWIN			m_pWin[DISPLAYTEXT_MAX_WINDOWS];

public:
//
// Deallocate the memory allocated in DisplayTextSetTextures(...)
//
void ReleaseTextures();

//
// Flushes all the print text commands
//
int Flush();

private:

#if __IPHONE_OS_VERSION_MAX_ALLOWED >= 30000
	// Declare the fragment and vertex shaders.
	GLuint uiFragShader, uiVertShader;		// Used to hold the fragment and vertex shader handles
	GLuint uiProgramObject;					// Used to hold the program handle (made out of the two previous shaders)
	// Handles for the uniform variables.
	int PMVMatrixHandle;
	int TextureHandle;
#endif
	
//
// Initialization and texture upload. Should be called only once
// for a given context.
//
bool APIInit();
void APIRelease();
	
//
// Initialization and texture upload. Should be called only once
// for a given context.
//
bool APIUpLoadIcons(
		const unsigned long * const pPVR,
		const unsigned long * const pIMG);

//
// true if succesful, false otherwise.
// Reads texture data from *.dat and loads it in
// video memory.
//
bool APIUpLoad4444(unsigned int TexID, unsigned char *pSource, unsigned int nSize, unsigned int nMode);

void DrawBackgroundWindowUP(unsigned int dwWin, SDisplayTextAPIVertex *pVtx, const bool bIsOp, const bool bBorder);

//
// Stores, writes and restores Render States
//
void APIRenderStates(int nAction);

//
// nPos = -1 to the left
// nPos = +1 to the right
//
void APIDrawLogo(unsigned int uLogoToDisplay, int nPos);
};


#endif