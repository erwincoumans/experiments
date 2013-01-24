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
#ifndef _TEXTURE_h_
#define _TEXTURE_h_

#include <TargetConditionals.h>
#include <Availability.h>
#import <OpenGLES/EAGL.h>
#if __IPHONE_OS_VERSION_MAX_ALLOWED >= 30000
#import <OpenGLES/ES2/gl.h>
#import <OpenGLES/ES2/glext.h>
#else
#import <OpenGLES/ES1/gl.h>
#import <OpenGLES/ES1/glext.h>
#endif
#include <stdio.h>

#include "MemoryManager.h"
#include "OpenGLESExt.h"


//#include "String.h"

// Describes the header of a PVR header-texture
typedef struct PVR_Header_Texture_TAG
{
	unsigned int dwHeaderSize;			/*!< size of the structure */
	unsigned int dwHeight;				/*!< height of surface to be created */
	unsigned int dwWidth;				/*!< width of input surface */
	unsigned int dwMipMapCount;			/*!< number of mip-map levels requested */
	unsigned int dwpfFlags;				/*!< pixel format flags */
	unsigned int dwTextureDataSize;		/*!< Total size in bytes */
	unsigned int dwBitCount;			/*!< number of bits per pixel  */
	unsigned int dwRBitMask;			/*!< mask for red bit */
	unsigned int dwGBitMask;			/*!< mask for green bits */
	unsigned int dwBBitMask;			/*!< mask for blue bits */
	unsigned int dwAlphaBitMask;		/*!< mask for alpha channel */
	unsigned int dwPVR;					/*!< magic number identifying pvr file */
	unsigned int dwNumSurfs;			/*!< the number of surfaces present in the pvr */
} PVR_Texture_Header;


typedef enum iPixelType_TAG
{
	MGLPT_ARGB_4444 = 0x00,
	MGLPT_ARGB_1555,
	MGLPT_RGB_565,
	MGLPT_RGB_555,
	MGLPT_RGB_888,
	MGLPT_ARGB_8888,
	MGLPT_ARGB_8332,
	MGLPT_I_8,
	MGLPT_AI_88,
	MGLPT_1_BPP,
	MGLPT_VY1UY0,
	MGLPT_Y1VY0U,
	MGLPT_PVRTC2,
	MGLPT_PVRTC4,
	MGLPT_PVRTC2_2,
	MGLPT_PVRTC2_4,

	OGL_RGBA_4444= 0x10,
	OGL_RGBA_5551,
	OGL_RGBA_8888,
	OGL_RGB_565,
	OGL_RGB_555,
	OGL_RGB_888,
	OGL_I_8,
	OGL_AI_88,
	OGL_PVRTC2,
	OGL_PVRTC4,
	OGL_PVRTC2_2,
	OGL_PVRTC2_4,

	D3D_DXT1 = 0x20,
	D3D_DXT2,
	D3D_DXT3,
	D3D_DXT4,
	D3D_DXT5,

	D3D_RGB_332,
	D3D_AI_44,
	D3D_LVU_655,
	D3D_XLVU_8888,
	D3D_QWVU_8888,

	//10 bits per channel
	D3D_ABGR_2101010,
	D3D_ARGB_2101010,
	D3D_AWVU_2101010,

	//16 bits per channel
	D3D_GR_1616,
	D3D_VU_1616,
	D3D_ABGR_16161616,

	//HDR formats
	D3D_R16F,
	D3D_GR_1616F,
	D3D_ABGR_16161616F,

	//32 bits per channel
	D3D_R32F,
	D3D_GR_3232F,
	D3D_ABGR_32323232F,

	// Ericsson
	ETC_RGB_4BPP,
	ETC_RGBA_EXPLICIT,
	ETC_RGBA_INTERPOLATED,

	MGLPT_NOTYPE = 0xff

} iPixelType;

const unsigned int PVRTEX_MIPMAP		= (1<<8);		// has mip map levels
const unsigned int PVRTEX_TWIDDLE		= (1<<9);		// is twiddled
const unsigned int PVRTEX_BUMPMAP		= (1<<10);		// has normals encoded for a bump map
const unsigned int PVRTEX_TILING		= (1<<11);		// is bordered for tiled pvr
const unsigned int PVRTEX_CUBEMAP		= (1<<12);		// is a cubemap/skybox
const unsigned int PVRTEX_FALSEMIPCOL	= (1<<13);		//
const unsigned int PVRTEX_VOLUME		= (1<<14);
const unsigned int PVRTEX_PIXELTYPE		= 0xff;			// pixel type is always in the last 16bits of the flags
const unsigned int PVRTEX_IDENTIFIER	= 0x21525650;	// the pvr identifier is the characters 'P','V','R'

const unsigned int PVRTEX_V1_HEADER_SIZE = 44;			// old header size was 44 for identification purposes

const unsigned int PVRTC2_MIN_TEXWIDTH		= 16;
const unsigned int PVRTC2_MIN_TEXHEIGHT		= 8;
const unsigned int PVRTC4_MIN_TEXWIDTH		= 8;
const unsigned int PVRTC4_MIN_TEXHEIGHT		= 8;
const unsigned int ETC_MIN_TEXWIDTH			= 4;
const unsigned int ETC_MIN_TEXHEIGHT		= 4;


class CTexture
{
public:
	CTexture();
	~CTexture();
//
// w - Size of the texture
// h - Size of the texture
// wMin - Minimum size of a texture level
// hMin - Minimum size of a texture level
// nBPP - Bits per pixel of the format
// bMIPMap - Create memory for MIP-map levels also?
//
// Allocated texture memory (must be free()d)
// Creates a PVR_Texture_Header structure, including room for the specified texture, in memory.
PVR_Texture_Header *TextureCreate(
									unsigned int		w,
									unsigned int		h,
									const unsigned int	wMin,
									const unsigned int	hMin,
									const unsigned int	nBPP,
									const bool			bMIPMap);
//
// pOut - The tiled texture in system memory
// pIn - The source texture
// nRepeatCnt - Number of times to repeat the source texture
// Allocates and fills, in system memory, a texture large enough to repeat the source texture specified number of times.
void TextureTile(
				PVR_Texture_Header			**pOut,
				const PVR_Texture_Header	* const pIn,
				const int					nRepeatCnt);

//
// Needed by PVRTTextureTile() in the various PVRTTextureAPIs
//
void TextureLoadTiled(
					unsigned char		* const pDst,
					const unsigned int	nWidthDst,
					const unsigned int	nHeightDst,
					const unsigned char	* const pSrc,
					const unsigned int	nWidthSrc,
					const unsigned int	nHeightSrc,
					const unsigned int	nElementSize,		// Bytes per pixel
					const bool			bTwiddled);



//-------------------- load Texture from header -----------------------



//
// Loads the whole texture. Release texture by calling ReleaseTexture(). Decompresses 
// to RGBA8888 internally
//
unsigned int LoadDecompressedTextureFromPointer(const void* pointer, GLuint *texName, const void *psTextureHeader);

//
// Can load parts of a mipmapped texture (ie skipping the highest detailed levels).
// Release texture by calling PVRTReleaseTexture.  Decompresses to RGBA8888 internally.
//
// nLoadFromLevel	Which mipmap level to start loading from (0=all)
//
unsigned int LoadDecompressedPartialTextureFromPointer(const void *pointer,
														   unsigned int nLoadFromLevel,
														   GLuint *texName,
														   const void *psTextureHeader=NULL);

//														   
// Can load parts of a mipmaped texture (ie skipping the highest detailed levels).
// Release texture by calling ReleaseTexture.  Decompresses to RGBA8888 internally.
//
// altHeader		If null, texture follows header, else texture is here.
// nLoadFromLevel	Which mipmap level to start loading from (0=all)
//
unsigned int LoadPartialTextureFromPointer(const void * const pointer,
											   const void * const texPtr,
											   const unsigned int nLoadFromLevel,
											   GLuint * const texName,
											   const void *psTextureHeader=NULL);

//														   
// Loads the whole texture.
// Release texture by calling ReleaseTexture.  Decompresses to RGBA8888 internally.
//
unsigned int LoadTextureFromPointer(const void* pointer, GLuint *const texName, const void *psTextureHeader=NULL);
	
//-------------------- load Texture from PNG or JPG -------------------
unsigned int LoadTextureFromImageFile(const char * const filename, GLuint * const texName, const void *psTextureHeader=NULL);

//-------------------- load Texture from PVR -----------------------

unsigned int  LoadTextureFromPVR(const char * const filename, GLuint * const texName, const void *psTextureHeader=NULL);

//														   
// Can load parts of a mipmaped texture (ie skipping the highest detailed levels) from a PVR file.
// Release texture by calling ReleaseTexture.  
//
// altHeader		If null, texture follows header, else texture is here.
// nLoadFromLevel	Which mipmap level to start loading from (0=all)
//
unsigned int LoadPartialTextureFromPVR(const char * const filename,
										   const char * const altHeader,
										   const unsigned int nLoadFromLevel,
										   GLuint * const texName,
										   const void *psTextureHeader=NULL);

//
// Can load parts of a mipmapped texture (ie skipping the highest detailed levels) from a PVR file.
// Release texture by calling PVRTReleaseTexture.  Decompresses to RGBA8888 internally.
//
// nLoadFromLevel	Which mipmap level to start loading from (0=all)
//
unsigned int LoadDecompressedPartialTextureFromPVR(const char* const filename,
													   unsigned int nLoadFromLevel,
													   GLuint *texName,
													   const void *psTextureHeader=NULL);

//
// Loads the whole texture from a PVR file. Release texture by calling ReleaseTexture(). Decompresses 
// to RGBA8888 internally
//
unsigned int LoadDecompressedTextureFromPVR(const char* const filename, GLuint *const texName, const void *psTextureHeader);



//-------------------- Decompression -----------------------

void PVRTCDecompress(const void *pCompressedData,
				const int Do2bitMode,
				const int XDim,
				const int YDim,
				unsigned char* pResultImage);
				
int ETCDecompress(const void * const pSrcData,
						 const unsigned int &x,
						 const unsigned int &y,
						 void *pDestData,
						 const int &nMode);

//-------------------- Utility -----------------------

//
// Releases the resources used by a texture
//
void ReleaseTexture(GLuint texName);

//
// Returns the bits per pixel (BPP) of the format.
//
unsigned int TextureFormatBPP(const GLuint nFormat, const GLuint nType);

//private:
//    DECLARE_HEAP;
};

#endif // end of _TEXTURE_H_

