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
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include "Texture.h"
#include "OpenGLESExt.h"
#include "Pathes.h"
#include "ResourceFile.h"
#include <CoreGraphics/CoreGraphics.h>
#include <UIKit/UIKit.h>

//#include "Log.h"
#include "Resource.h"
#include "Macros.h"

extern int __OPENGLES_VERSION;

typedef unsigned long U32;
typedef unsigned char U8;

//DEFINE_HEAP(CTexture, "Textures");

CTexture::CTexture(){};
CTexture::~CTexture(){};

//
// filename			Filename of the .PNG or .JPG file to load the texture from
// texName			the OpenGL ES texture name as returned by glBindTexture
// psTextureHeader	Pointer to a PVR_Texture_Header struct. Modified to
//					contain the header data of the returned texture Ignored if NULL.
unsigned int CTexture::LoadTextureFromImageFile(const char * const filename, GLuint * const texName, const void *psTextureHeader) 
{
	CGImageRef textureImage;
	CGContextRef textureContext;
	
	GLuint textureID = 0;
	char buffer[2048];
	GetResourcePathASCII(buffer, 2048);
	strcat( buffer, filename );
	NSString *path = [NSString stringWithUTF8String:buffer];
	
	
	UIImage *uiImage = [UIImage imageWithContentsOfFile:path];
	if( uiImage ) {
		textureImage = uiImage.CGImage;
		
		int width = CGImageGetWidth(textureImage);
		int height = CGImageGetHeight(textureImage);
		
		GLubyte *textureData;
		
		if(textureImage) {
			textureData = (GLubyte *) malloc(width * height * 4);
			textureContext = CGBitmapContextCreate(textureData, width, height, 8, width * 4, CGImageGetColorSpace(textureImage), kCGImageAlphaPremultipliedLast);
			CGContextDrawImage(textureContext, CGRectMake(0.0, 0.0, (float)width, (float)height), textureImage);
		}
		//glActiveTexture(GL_TEXTURE0);
		glGenTextures(1, &textureID);
		glBindTexture(GL_TEXTURE_2D, textureID);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		if( __OPENGLES_VERSION < 2 ) {
			glTexParameteri(GL_TEXTURE_2D, GL_GENERATE_MIPMAP, GL_TRUE );
		}
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, textureData);
#if __IPHONE_OS_VERSION_MAX_ALLOWED >= 30000
		if( __OPENGLES_VERSION >= 2 ) {
			glGenerateMipmap( GL_TEXTURE_2D );
		}
#endif
		if(glGetError())
		{
			printf("glTexImage2D failed. ");
			//return 0;
		}
		free(textureData);
		CGContextRelease(textureContext);
	}
	else {
		GLubyte pixels[4*3] = {
			255, 0, 0, 
			0, 255, 0, 
			0, 0, 255,
			255,255,0
		};
		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
		glGenTextures(1, &textureID);
		glBindTexture(GL_TEXTURE_2D, textureID);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 2, 2, 0, GL_RGB, GL_UNSIGNED_BYTE, pixels);
	}

		
	*texName = textureID;
	
	return true;
}

//
//-------------------- load Texture from PVR -----------------------
//
//
// filename			Filename of the .PVR file to load the texture from
// texPtr			If null, texture follows header, else texture is here.
// nLoadFromLevel	Which mipmap level to start loading from (0=all)
// texName			the OpenGL ES texture name as returned by glBindTexture
// psTextureHeader	Pointer to a PVR_Texture_Header struct. Modified to
// 					contain the header data of the returned texture Ignored if NULL.	
//
// Allows textures to be stored in binary PVR files and loaded in. Can load parts of a
// mipmaped texture (ie skipping the highest detailed levels). Release texture by
// calling  ReleaseTexture.
//
unsigned int  CTexture::LoadPartialTextureFromPVR(const char * const filename,
										   const char * const texPtr,
										   const unsigned int nLoadFromLevel,
										   GLuint * const texName,
										   const void *psTextureHeader)
{
	CPVRTResourceFile TexFile(filename);
	if (!TexFile.IsOpen()) return 0;

	return  LoadPartialTextureFromPointer(TexFile.DataPtr(),
		texPtr,
		nLoadFromLevel,
		texName,
		psTextureHeader);		 
}

//
// filename			Filename of the .PVR file to load the texture from
// texName			the OpenGL ES texture name as returned by glBindTexture
// psTextureHeader	Pointer to a PVR_Texture_Header struct. Modified to
//					contain the header data of the returned texture Ignored if NULL.	
// 
// Allows textures to be stored in binary PVR files and loaded in. Loads the whole texture
// Release texture by calling  ReleaseTexture.
// 
unsigned int  CTexture::LoadTextureFromPVR(const char * const filename, GLuint * const texName, const void *psTextureHeader)
{
	return  LoadPartialTextureFromPVR(filename, 0, 0, texName, psTextureHeader);
}

// 
// filename		Filename of the .PVR file to load the texture from
// nLoadFromLevel	Which mipmap level to start loading from (0=all)
// texName			the OpenGL ES texture name as returned by glBindTexture
// psTextureHeader	Pointer to a PVR_Texture_Header struct. Modified to
//					contain the header data of the returned texture Ignored if NULL.	
// 
// Allows textures to be stored in binary PVR files and loaded in. Can load parts of a
// mipmaped texture (ie skipping the highest detailed levels). Release texture by
// calling  ReleaseTexture.  This variant decompresses to RGBA8888.
// 
unsigned int  CTexture::LoadDecompressedPartialTextureFromPVR(const char* const filename, unsigned int nLoadFromLevel, GLuint *const texName, const void *psTextureHeader)
{
	CPVRTResourceFile TexFile(filename);
	if (!TexFile.IsOpen()) return 0;

	return  LoadDecompressedPartialTextureFromPointer(TexFile.DataPtr(),
		nLoadFromLevel,
		texName,
		psTextureHeader);
}

// filename		Filename of the .PVR file to load the texture from
// texName			the OpenGL ES texture name as returned by glBindTexture
// psTextureHeader	Pointer to a PVR_Texture_Header struct. Modified to
// 					contain the header data of the returned texture Ignored if NULL.	
// 
// Allows textures to be stored in binary PVR files and loaded in. Loads the whole texture
// Release texture by calling  ReleaseTexture.  This variant decompresses to RGBA8888.
// 
unsigned int  CTexture::LoadDecompressedTextureFromPVR(const char* const filename, GLuint *texName, const void *psTextureHeader)
{
	return  LoadDecompressedPartialTextureFromPVR(filename, 0, texName, psTextureHeader);
}

//-------------------- load Texture from header -----------------------

// 
// texName			the OpenGL ES texture name as returned by glBindTexture
// pointer			Pointer to the header-prefixed texture from
// psTextureHeader	Pointer to a PVR_Texture_Header struct. Modified to
//					contain the header data of the returned texture Ignored if NULL.	
// true on success
// Allows textures to be stored in header files and loaded in. Loads the whole texture
// Release texture by calling  ReleaseTexture.  Decompresses to RGBA8888 internally.
// 
unsigned int  CTexture::LoadDecompressedTextureFromPointer(const void* pointer, GLuint *texName, const void *psTextureHeader)
{
	return  LoadDecompressedPartialTextureFromPointer(pointer, 0, texName, psTextureHeader);
}
// 
// pointer			Pointer to the header-prefixed texture from
// nLoadFromLevel	Which mipmap level to start loading from (0=all)
// texName			the OpenGL ES texture name as returned by glBindTexture
// psTextureHeader	Pointer to a PVR_Texture_Header struct. Modified to
//					contain the header data of the returned texture Ignored if NULL.	
// 
// Allows textures to be stored in header files and loaded in. Loads the whole texture
// Release texture by calling  ReleaseTexture.  Decompresses to RGBA8888 internally.
// 
unsigned int  CTexture::LoadDecompressedPartialTextureFromPointer(const void *pointer,
														   unsigned int nLoadFromLevel,
														   GLuint *texName,
														   const void *psTextureHeader)
{
	// new off a duplicate pointer for the header _only_
	PVR_Texture_Header *oldHeader = (PVR_Texture_Header*)pointer;

	if(((oldHeader->dwpfFlags&PVRTEX_PIXELTYPE) != OGL_PVRTC2) && ((oldHeader->dwpfFlags&PVRTEX_PIXELTYPE) != OGL_PVRTC4))
	{
		return  LoadPartialTextureFromPointer(pointer, 0, 0, texName, psTextureHeader);
	}

	PVR_Texture_Header *newHeader = (PVR_Texture_Header*) malloc(sizeof(PVR_Texture_Header));
	memset(newHeader, 0, sizeof(PVR_Texture_Header));
	memcpy(newHeader,pointer,sizeof(PVR_Texture_Header));

	//Change the decompressed texture header's format to be RGBA8888, drop top mip level
	newHeader->dwpfFlags = OGL_RGBA_8888;
	newHeader->dwpfFlags |= (oldHeader->dwpfFlags & PVRTEX_MIPMAP);
	newHeader->dwMipMapCount--;
	newHeader->dwBitCount = 32;
	newHeader->dwWidth /= 2;
	newHeader->dwHeight /= 2;
	int newSize = 0;

	// new width*height*miplevels*4 bytes
	int szx = newHeader->dwWidth;
	int szy = newHeader->dwHeight;
	int nMips = 1;
	while(szx && szy){
		newSize += 4*szx*szy;
		szx>>=1;
		szy>>=1;
		nMips++;
	}

	unsigned char *newTexture = (unsigned char*) malloc(newSize * sizeof(unsigned char));
	memset(newTexture, 0, newSize * sizeof(unsigned char));

	//Decompress each texture layer into the new memory
	szx = newHeader->dwWidth;
	szy = newHeader->dwHeight;
	unsigned char *thisMipLevelSrc = (unsigned char*)pointer+sizeof(PVR_Texture_Header)+
		(oldHeader->dwWidth*oldHeader->dwHeight*oldHeader->dwBitCount + 7)/8;
	unsigned char *thisMipLevelDest = newTexture;
	while(szx && szy){
		 PVRTCDecompress(thisMipLevelSrc,
				((oldHeader->dwpfFlags&PVRTEX_PIXELTYPE) == OGL_PVRTC2) ? 1 : 0,
				szx,
				szy,
				thisMipLevelDest);
		thisMipLevelSrc += (szx * szy * oldHeader->dwBitCount + 7)/8;
		thisMipLevelDest += (szx * szy * 4);
		szx>>=1;
		szy>>=1;
	}

	//Load the textures
	unsigned int result =  LoadPartialTextureFromPointer(newHeader,newTexture, 0, texName, psTextureHeader);

	// make sure new header is passed back rather than file's header
	*(PVR_Texture_Header*)(psTextureHeader) = *newHeader;

	//Bin the temporary header and memory
	free(newHeader); 
	free(newTexture);

	return result;
}


//														   
// Loads the whole texture.
// Release texture by calling ReleaseTexture.  Decompresses to RGBA8888 internally.
//
//
//	In a OpenGL Cube Map, each texture's up direction is defined as next (view direction, up direction),
//	(+x,-y)(-x,-y)(+y,+z)(-y,-z)(+z,-y)(-z,-y).
//
unsigned int CTexture::LoadPartialTextureFromPointer(const void * const pointer,
											   const void * const texPtr,
											   const unsigned int nLoadFromLevel,
											   GLuint * const texName,
											   const void *psTextureHeader)
{
	PVR_Texture_Header* psPVRHeader = (PVR_Texture_Header*)pointer;
	unsigned int u32NumSurfs;

	// perform checks for old PVR psPVRHeader
	if(psPVRHeader->dwHeaderSize!=sizeof(PVR_Texture_Header))
	{	// Header V1
		if(psPVRHeader->dwHeaderSize==PVRTEX_V1_HEADER_SIZE)
		{	// react to old psPVRHeader: i.e. fill in numsurfs as this is missing from old header
			printf("LoadPartialTextureFromPointer warning: this is an old pvr"
				" - you can use PVRTexTool to update its header.");
			if(psPVRHeader->dwpfFlags&PVRTEX_CUBEMAP)
				u32NumSurfs = 6;
			else
				u32NumSurfs = 1;
		}
		else
		{	// not a pvr at all
			printf("LoadPartialTextureFromPointer failed: not a valid pvr. ");
			return 0;
		}
	}
	else
	{	// Header V2
		if(psPVRHeader->dwNumSurfs<1)
		{	// encoded with old version of PVRTexTool before zero numsurfs bug found.
			if(psPVRHeader->dwpfFlags & PVRTEX_CUBEMAP)
				u32NumSurfs = 6;
			else
				u32NumSurfs = 1;
		}
		else
		{
			u32NumSurfs = psPVRHeader->dwNumSurfs;
		}
	}

	GLuint textureName;
	GLenum textureFormat = 0;
	GLenum textureType = GL_RGB;
	GLenum eTarget; //OGLES2 Variable.

	bool IsPVRTCSupported = true; // hack until extension check is available //OpenGLESExt::IsGLExtensionSupported("GL_IMG_texture_compression_pvrtc");

	*texName = 0;	// install warning value
	bool IsCompressedFormatSupported = false, IsCompressedFormat = false;

	/* Only accept untwiddled data UNLESS texture format is PVRTC */
	if ( ((psPVRHeader->dwpfFlags & PVRTEX_TWIDDLE) == PVRTEX_TWIDDLE)
		&& ((psPVRHeader->dwpfFlags & PVRTEX_PIXELTYPE)!=OGL_PVRTC2)
		&& ((psPVRHeader->dwpfFlags & PVRTEX_PIXELTYPE)!=OGL_PVRTC4) )
	{
		// We need to load untwiddled textures -- hw will twiddle for us.
		printf("LoadPartialTextureFromPointer failed: texture should be untwiddled. ");
		return 0;
	}

	switch(psPVRHeader->dwpfFlags & PVRTEX_PIXELTYPE)
	{
	case OGL_RGBA_4444:
		textureFormat = GL_UNSIGNED_SHORT_4_4_4_4;
		textureType = GL_RGBA;
		break;

	case OGL_RGBA_5551:
		textureFormat = GL_UNSIGNED_SHORT_5_5_5_1;
		textureType = GL_RGBA;
		break;

	case OGL_RGBA_8888:
		textureFormat = GL_UNSIGNED_BYTE;
		textureType = GL_RGBA;
		break;

	/* New OGL Specific Formats Added */

	case OGL_RGB_565:
		textureFormat = GL_UNSIGNED_SHORT_5_6_5;
		textureType = GL_RGB;
		break;

	case OGL_RGB_555:
		printf("LoadPartialTextureFromPointer failed: pixel type OGL_RGB_555 not supported. ");
		return 0; // Deal with exceptional case

	case OGL_RGB_888:
		textureFormat = GL_UNSIGNED_BYTE;
		textureType = GL_RGB;
		break;

	case OGL_I_8:
		textureFormat = GL_UNSIGNED_BYTE;
		textureType = GL_LUMINANCE;
		break;

	case OGL_AI_88:
		textureFormat = GL_UNSIGNED_BYTE;
		textureType = GL_LUMINANCE_ALPHA;
		break;

	case OGL_PVRTC2:
		if(IsPVRTCSupported)
		{
			IsCompressedFormatSupported = IsCompressedFormat = true;
			textureFormat = psPVRHeader->dwAlphaBitMask==0 ? GL_COMPRESSED_RGB_PVRTC_2BPPV1_IMG : GL_COMPRESSED_RGBA_PVRTC_2BPPV1_IMG ;	// PVRTC2
		}
		else
		{
			IsCompressedFormatSupported = false;
			IsCompressedFormat = true;
			textureFormat = GL_UNSIGNED_BYTE;
			textureType = GL_RGBA;
			printf("LoadPartialTextureFromPointer warning: PVRTC2 not supported. Converting to RGBA8888 instead. ");
		}
		break;

	case OGL_PVRTC4:
		if(IsPVRTCSupported)
		{
			IsCompressedFormatSupported = IsCompressedFormat = true;
			textureFormat = psPVRHeader->dwAlphaBitMask==0 ? GL_COMPRESSED_RGB_PVRTC_4BPPV1_IMG : GL_COMPRESSED_RGBA_PVRTC_4BPPV1_IMG ;	// PVRTC4
		}
		else
		{
			IsCompressedFormatSupported = false;
			IsCompressedFormat = true;
			textureFormat = GL_UNSIGNED_BYTE;
			textureType = GL_RGBA;
			printf("LoadPartialTextureFromPointer warning: PVRTC4 not supported. Converting to RGBA8888 instead. ");
		}
		break;

	default:											// NOT SUPPORTED
		printf("LoadPartialTextureFromPointer failed: pixel type not supported. ");
		return 0;
	}

	// load the texture up
	glPixelStorei(GL_UNPACK_ALIGNMENT,1);				// Never have row-aligned in psPVRHeaders

	glGenTextures(1, &textureName);

	//  check that this data is cube map data or not.
	if(psPVRHeader->dwpfFlags & PVRTEX_CUBEMAP)
	{ 		
#if __IPHONE_OS_VERSION_MAX_ALLOWED >= 30000
		if( __OPENGLES_VERSION >= 2 ) 
		{
			// OGLES 2.0, let's go ahead and make the eTarget a Cube Map.
			eTarget = GL_TEXTURE_CUBE_MAP;
			glBindTexture(GL_TEXTURE_CUBE_MAP, textureName);
		}
		else 
#endif
		{
			// Not in OGLES1.1, you don't!
			printf("LoadPartialTextureFromPointer failed: cube map textures are not available in OGLES1.x. ");
			return 0;
		}
	}
	else
	{
		eTarget = GL_TEXTURE_2D;
		glBindTexture(GL_TEXTURE_2D, textureName);
	}

	if(glGetError())
	{
		printf("LoadPartialTextureFromPointer failed: glBindTexture() failed. ");
		return 0;
	}

	for(unsigned int i=0; i<u32NumSurfs; i++)
	{
		char *theTexturePtr = (texPtr? (char*)texPtr :  (char*)psPVRHeader + psPVRHeader->dwHeaderSize) + psPVRHeader->dwTextureDataSize * i;
		char *theTextureToLoad = 0;
		int		nMIPMapLevel;
		int		nTextureLevelsNeeded = (psPVRHeader->dwpfFlags & PVRTEX_MIPMAP)? psPVRHeader->dwMipMapCount : 0;
		unsigned int		nSizeX= psPVRHeader->dwWidth, nSizeY = psPVRHeader->dwHeight;
		unsigned int		CompressedImageSize = 0;

		for(nMIPMapLevel = 0; nMIPMapLevel <= nTextureLevelsNeeded; nSizeX=_MAX(nSizeX/2, 1), nSizeY=_MAX(nSizeY/2, 1), nMIPMapLevel++)
		{
			// Do Alpha-swap if needed

			theTextureToLoad = theTexturePtr;

			// Load the Texture

			/* If the texture is PVRTC then use GLCompressedTexImage2D */
			if(IsCompressedFormat)
			{
				/* Calculate how many bytes this MIP level occupies */
				if ((psPVRHeader->dwpfFlags & PVRTEX_PIXELTYPE)==OGL_PVRTC2)
				{
					CompressedImageSize = ( max(nSizeX, PVRTC2_MIN_TEXWIDTH) * max(nSizeY, PVRTC2_MIN_TEXHEIGHT) * psPVRHeader->dwBitCount + 7) / 8;
				}
				else
				{// PVRTC4 case
					CompressedImageSize = ( max(nSizeX, PVRTC4_MIN_TEXWIDTH) * max(nSizeY, PVRTC4_MIN_TEXHEIGHT) * psPVRHeader->dwBitCount + 7) / 8;
				}

				if(((signed int)nMIPMapLevel - (signed int)nLoadFromLevel) >= 0)
				{
					if(IsCompressedFormatSupported)
					{
						if(psPVRHeader->dwpfFlags&PVRTEX_CUBEMAP)
						{
							#if __IPHONE_OS_VERSION_MAX_ALLOWED >= 30000
							if( __OPENGLES_VERSION >= 2 ) 
							{
								//And should enter here.
								/* Load compressed texture data at selected MIP level */
								glCompressedTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X+i, nMIPMapLevel-nLoadFromLevel, textureFormat, nSizeX, nSizeY, 0,
													   CompressedImageSize, theTextureToLoad);
							}
							else 
							#endif
							{
								printf("Should not reach this statement.  Trying to load a cubemap for OpenGL ES 1.1.\n");
								return 0;
							}
						}
						else
						{
							/* Load compressed texture data at selected MIP level */
							glCompressedTexImage2D(GL_TEXTURE_2D, nMIPMapLevel-nLoadFromLevel, textureFormat, nSizeX, nSizeY, 0,
												   CompressedImageSize, theTextureToLoad);
							
						}
					}
					else
					{
						// Convert PVRTC to 32-bit
						unsigned char *u8TempTexture = (unsigned char*) malloc(nSizeX*nSizeY*4 * sizeof(unsigned char));
						memset(u8TempTexture, 0, nSizeX*nSizeY*4 * sizeof(unsigned char));
						
						if ((psPVRHeader->dwpfFlags & PVRTEX_PIXELTYPE)==OGL_PVRTC2)
						{
							PVRTCDecompress(theTextureToLoad, 1, nSizeX, nSizeY, u8TempTexture);
						}
						else
						{// PVRTC4 case
							PVRTCDecompress(theTextureToLoad, 0, nSizeX, nSizeY, u8TempTexture);
						}


						if(psPVRHeader->dwpfFlags&PVRTEX_CUBEMAP)
						{
#if __IPHONE_OS_VERSION_MAX_ALLOWED >= 30000
							if( __OPENGLES_VERSION >= 2 ) 
							{
								// Load compressed cubemap data at selected MIP level
								// Upload the texture as 32-bits
								glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X+i,nMIPMapLevel-nLoadFromLevel,GL_RGBA,
											 nSizeX,nSizeY,0, GL_RGBA,GL_UNSIGNED_BYTE,u8TempTexture);
								free(u8TempTexture);
							}
							else 
#endif
							{
								printf("Failed when trying to load an unsupported feature (cubemap) for OpenGL ES 1.1.\n");
								return 0;
							}
						}
						else
						{// Load compressed 2D data at selected MIP level
							// Upload the texture as 32-bits
							glTexImage2D(GL_TEXTURE_2D,nMIPMapLevel-nLoadFromLevel,GL_RGBA,
										 nSizeX,nSizeY,0, GL_RGBA,GL_UNSIGNED_BYTE,u8TempTexture);
							free(u8TempTexture);
						}
					}
				}
			}
			else
			{
				if(((signed int)nMIPMapLevel - (signed int)nLoadFromLevel) >= 0)
				{
					if(psPVRHeader->dwpfFlags&PVRTEX_CUBEMAP)
					{
#if __IPHONE_OS_VERSION_MAX_ALLOWED >= 30000
						if( __OPENGLES_VERSION >= 2 ) 
						{
							/* Load uncompressed texture data at selected MIP level */
							glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X+i,nMIPMapLevel-nLoadFromLevel,textureType,nSizeX,nSizeY,
										 0, textureType,textureFormat,theTextureToLoad);
						}
						else 
#endif
						{
							printf("Failed when trying to load an unsupported feature (cubemap) for OpenGL ES 1.1.\n");
							return 0;
						}
					}
					else
					{
						/* Load uncompressed texture data at selected MIP level */
						glTexImage2D(GL_TEXTURE_2D,nMIPMapLevel-nLoadFromLevel,textureType,nSizeX,nSizeY,0, textureType,textureFormat,theTextureToLoad);
					}
				}
			}

			if(glGetError())
			{
				printf("LoadPartialTextureFromPointer failed: glBindTexture() failed. ");
				return 0;
			}

			// offset the texture pointer by one mip-map level

			/* PVRTC case */
			if ( IsCompressedFormat )
			{
				theTexturePtr += CompressedImageSize;
			}
			else
			{
				/* New formula that takes into account bit counts inferior to 8 (e.g. 1 bpp) */
				theTexturePtr += (nSizeX * nSizeY * psPVRHeader->dwBitCount + 7) / 8;
			}
		}
	}

	*texName = textureName;

	if(psTextureHeader)
	{
		*(PVR_Texture_Header*)psTextureHeader = *psPVRHeader;
		((PVR_Texture_Header*)psTextureHeader)->dwPVR = PVRTEX_IDENTIFIER;
		((PVR_Texture_Header*)psTextureHeader)->dwNumSurfs = u32NumSurfs;
	}

	/* 1.1 Return line. */
	//return psPVRHeader->dwpfFlags|0x80000000;		// PVR psPVRHeader flags with topmost bit set so that it is non-zero
	
	/* 2.0 Return line. */
	if(!psPVRHeader->dwMipMapCount)
	{
		glTexParameteri(eTarget, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(eTarget, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	}
	else
	{
        glTexParameteri(eTarget, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST);
		glTexParameteri(eTarget, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	}
	
	return 1; 
}


//
//
//
unsigned int CTexture::LoadTextureFromPointer(const void* pointer, GLuint *const texName, const void *psTextureHeader)
{
	return LoadPartialTextureFromPointer(pointer, 0, 0, texName, psTextureHeader);
}


//
// release OpenGL resources
//
void CTexture::ReleaseTexture(GLuint texName)
{
	glDeleteTextures(1,&texName);
}








//
//
//------------------------------ Decompressing Textures --------------------------------------
//
//

//
//
//
typedef struct
{
	/* Uses 64 bits pre block*/
	U32 PackedData[2];
}AMTC_BLOCK_STRUCT;


static void Decompress(AMTC_BLOCK_STRUCT *pCompressedData,
					   const int Do2bitMode,
					   const int XDim,
					   const int YDim,
					   const int AssumeImageTiles,
					   unsigned char* pResultImage);

void CTexture::PVRTCDecompress(const void *pCompressedData,
				const int Do2bitMode,
				const int XDim,
				const int YDim,
				unsigned char* pResultImage)
{
	Decompress((AMTC_BLOCK_STRUCT*)pCompressedData,Do2bitMode,XDim,YDim,1,pResultImage);
}


/******************************************************************************
 * Function Name: util_number_is_power_2
 *
 * Inputs       : input - A number.
 * Outputs      : -
 * Returns      : TRUE if the number is an integer power of two, else FALSE.
 * Globals Used : -
 *
 * Description  : Check that a number is an integer power of two, i.e.
 *                1, 2, 4, 8, ... etc.
 *                Returns FALSE for zero.
 * Pre-condition: -
 *****************************************************************************/
int util_number_is_power_2( unsigned  input )
{
  unsigned minus1;

  if( !input ) return 0;

  minus1 = input - 1;
  return ( (input | minus1) == (input ^ minus1) ) ? 1 : 0;
}

/***********************************************************/
/*
// Unpack5554Colour
//
// Given a block, extract the colour information and convert to 5554 formats
*/
/***********************************************************/

static void Unpack5554Colour(const AMTC_BLOCK_STRUCT *pBlock,
							 int   ABColours[2][4])
{
	U32 RawBits[2];

	int i;

	/*
	// Extract A and B
	*/
	RawBits[0] = pBlock->PackedData[1] & (0xFFFE); /*15 bits (shifted up by one)*/
	RawBits[1] = pBlock->PackedData[1] >> 16;	   /*16 bits*/

	/*
	//step through both colours
	*/
	for(i = 0; i < 2; i++)
	{
		/*
		// if completely opaque
		*/
		if(RawBits[i] & (1<<15))
		{
			/*
			// Extract R and G (both 5 bit)
			*/
			ABColours[i][0] = (RawBits[i] >> 10) & 0x1F;
			ABColours[i][1] = (RawBits[i] >>  5) & 0x1F;

			/*
			// The precision of Blue depends on  A or B. If A then we need to
			// replicate the top bit to get 5 bits in total
			*/
			ABColours[i][2] = RawBits[i] & 0x1F;
			if(i==0)
			{
				ABColours[0][2] |= ABColours[0][2] >> 4;
			}

			/*
			// set 4bit alpha fully on...
			*/
			ABColours[i][3] = 0xF;
		}
		/*
		// Else if colour has variable translucency
		*/
		else
		{
			/*
			// Extract R and G (both 4 bit).
			// (Leave a space on the end for the replication of bits
			*/
			ABColours[i][0] = (RawBits[i] >>  (8-1)) & 0x1E;
			ABColours[i][1] = (RawBits[i] >>  (4-1)) & 0x1E;

			/*
			// replicate bits to truly expand to 5 bits
			*/
			ABColours[i][0] |= ABColours[i][0] >> 4;
			ABColours[i][1] |= ABColours[i][1] >> 4;

			/*
			// grab the 3(+padding) or 4 bits of blue and add an extra padding bit
			*/
			ABColours[i][2] = (RawBits[i] & 0xF) << 1;

			/*
			// expand from 3 to 5 bits if this is from colour A, or 4 to 5 bits if from
			// colour B
			*/
			if(i==0)
			{
				ABColours[0][2] |= ABColours[0][2] >> 3;
			}
			else
			{
				ABColours[0][2] |= ABColours[0][2] >> 4;
			}

			/*
			// Set the alpha bits to be 3 + a zero on the end
			*/
			ABColours[i][3] = (RawBits[i] >> 11) & 0xE;
		}/*end if variable alpha*/
	}/*end for i*/

}


/***********************************************************/
/*
// UnpackModulations
//
// Given the block and the texture type and it's relative position in the
// 2x2 group of blocks, extract the bit patterns for the fully defined pixels.
*/
/***********************************************************/

static void	UnpackModulations(const AMTC_BLOCK_STRUCT *pBlock,
							  const int Do2bitMode,
							  int ModulationVals[8][16],
							  int ModulationModes[8][16],
							  int StartX,
							  int StartY)
{
	int BlockModMode;
	U32 ModulationBits;

	int x, y;

	BlockModMode= pBlock->PackedData[1] & 1;
	ModulationBits	= pBlock->PackedData[0];

	/*
	// if it's in an interpolated mode
	*/
	if(Do2bitMode && BlockModMode)
	{
		/*
		// run through all the pixels in the block. Note we can now treat all the
		// "stored" values as if they have 2bits (even when they didn't!)
		*/
		for(y = 0; y < BLK_Y_SIZE; y++)
		{
			for(x = 0; x < BLK_X_2BPP; x++)
			{
				ModulationModes[y+StartY][x+StartX] = BlockModMode;

				/*
				// if this is a stored value...
				*/
				if(((x^y)&1) == 0)
				{
					ModulationVals[y+StartY][x+StartX] = ModulationBits & 3;
					ModulationBits >>= 2;
				}
			}
		}/*end for y*/
	}
	/*
	// else if direct encoded 2bit mode - i.e. 1 mode bit per pixel
	*/
	else if(Do2bitMode)
	{
		for(y = 0; y < BLK_Y_SIZE; y++)
		{
			for(x = 0; x < BLK_X_2BPP; x++)
			{
				ModulationModes[y+StartY][x+StartX] = BlockModMode;

				/*
				// double the bits so 0=> 00, and 1=>11
				*/
				if(ModulationBits & 1)
				{
					ModulationVals[y+StartY][x+StartX] = 0x3;
				}
				else
				{
					ModulationVals[y+StartY][x+StartX] = 0x0;
				}
				ModulationBits >>= 1;
			}
		}/*end for y*/
	}
	/*
	// else its the 4bpp mode so each value has 2 bits
	*/
	else
	{
		for(y = 0; y < BLK_Y_SIZE; y++)
		{
			for(x = 0; x < BLK_X_4BPP; x++)
			{
				ModulationModes[y+StartY][x+StartX] = BlockModMode;

				ModulationVals[y+StartY][x+StartX] = ModulationBits & 3;
				ModulationBits >>= 2;
			}
		}/*end for y*/
	}

	/*
	// make sure nothing is left over
	*/
	_ASSERT(ModulationBits==0);
}


/***********************************************************/
/*
// Interpolate Colours
//
//
// This performs a HW bit accurate interpolation of either the
// A or B colours for a particular pixel
//
// NOTE: It is assumed that the source colours are in ARGB 5554 format -
//		 This means that some "preparation" of the values will be necessary.
*/
/***********************************************************/

static void InterpolateColours(const int ColourP[4],
						  const int ColourQ[4],
						  const int ColourR[4],
						  const int ColourS[4],
						  const int Do2bitMode,
						  const int x,
						  const int y,
						  int Result[4])
{
	int u, v, uscale;
	int k;

	int tmp1, tmp2;

	int P[4], Q[4], R[4], S[4];

	/*
	// Copy the colours
	*/
	for(k = 0; k < 4; k++)
	{
		P[k] = ColourP[k];
		Q[k] = ColourQ[k];
		R[k] = ColourR[k];
		S[k] = ColourS[k];
	}

	/*
	// put the x and y values into the right range
	*/
	v = (y & 0x3) | ((~y & 0x2) << 1);
	if(Do2bitMode)
	{
		u = (x & 0x7) | ((~x & 0x4) << 1);

	}
	else
	{
		u = (x & 0x3) | ((~x & 0x2) << 1);
	}



	/*
	// get the u and v scale amounts
	*/
	v  = v - BLK_Y_SIZE/2;

	if(Do2bitMode)
	{
		u = u - BLK_X_2BPP/2;
		uscale = 8;
	}
	else
	{
		u = u - BLK_X_4BPP/2;
		uscale = 4;
	}

	for(k = 0; k < 4; k++)
	{
		tmp1 = P[k] * uscale + u * (Q[k] - P[k]);
		tmp2 = R[k] * uscale + u * (S[k] - R[k]);

		tmp1 = tmp1 * 4 + v * (tmp2 - tmp1);

		Result[k] = tmp1;
	}

	/*
	// Lop off the appropriate number of bits to get us to 8 bit precision
	*/
	if(Do2bitMode)
	{
		/*
		// do RGB
		*/
		for(k = 0; k < 3; k++)
		{
			Result[k] >>= 2;
		}

		Result[3] >>= 1;
	}
	else
	{
		/*
		// do RGB  (A is ok)
		*/
		for(k = 0; k < 3; k++)
		{
			Result[k] >>= 1;
		}
	}

	/*
	// sanity check
	*/
	for(k = 0; k < 4; k++)
	{
		_ASSERT(Result[k] < 256);
	}


	/*
	// Convert from 5554 to 8888
	//
	// do RGB 5.3 => 8
	*/
	for(k = 0; k < 3; k++)
	{
		Result[k] += Result[k] >> 5;
	}
	Result[3] += Result[3] >> 4;

	/*
	// 2nd sanity check
	*/
	for(k = 0; k < 4; k++)
	{
		_ASSERT(Result[k] < 256);
	}

}

/***********************************************************/
/*
// GetModulationValue
//
// Get the modulation value as a numerator of a fraction of 8ths
*/
/***********************************************************/
static void GetModulationValue(int x,
							   int y,
							   const int Do2bitMode,
							   const int ModulationVals[8][16],
							   const int ModulationModes[8][16],
							   int *Mod,
							   int *DoPT)
{
	static const int RepVals0[4] = {0, 3, 5, 8};
	static const int RepVals1[4] = {0, 4, 4, 8};

	int ModVal;

	/*
	// Map X and Y into the local 2x2 block
	*/
	y = (y & 0x3) | ((~y & 0x2) << 1);
	if(Do2bitMode)
	{
		x = (x & 0x7) | ((~x & 0x4) << 1);

	}
	else
	{
		x = (x & 0x3) | ((~x & 0x2) << 1);
	}

	/*
	// assume no PT for now
	*/
	*DoPT = 0;

	/*
	// extract the modulation value. If a simple encoding
	*/
	if(ModulationModes[y][x]==0)
	{
		ModVal = RepVals0[ModulationVals[y][x]];
	}
	else if(Do2bitMode)
	{
		/*
		// if this is a stored value
		*/
		if(((x^y)&1)==0)
		{
			ModVal = RepVals0[ModulationVals[y][x]];
		}
		/*
		// else average from the neighbours
		//
		// if H&V interpolation...
		*/
		else if(ModulationModes[y][x] == 1)
		{
			ModVal = (RepVals0[ModulationVals[y-1][x]] +
					  RepVals0[ModulationVals[y+1][x]] +
					  RepVals0[ModulationVals[y][x-1]] +
					  RepVals0[ModulationVals[y][x+1]] + 2) / 4;
		}
		/*
		// else if H-Only
		*/
		else if(ModulationModes[y][x] == 2)
		{
			ModVal = (RepVals0[ModulationVals[y][x-1]] +
					  RepVals0[ModulationVals[y][x+1]] + 1) / 2;
		}
		/*
		// else it's V-Only
		*/
		else
		{
			ModVal = (RepVals0[ModulationVals[y-1][x]] +
					  RepVals0[ModulationVals[y+1][x]] + 1) / 2;

		}/*end if/then/else*/
	}
	/*
	// else it's 4BPP and PT encoding
	*/
	else
	{
		ModVal = RepVals1[ModulationVals[y][x]];

		*DoPT = ModulationVals[y][x] == PT_INDEX;
	}

	*Mod =ModVal;
}


/*****************************************************************************/
/*
// TwiddleUV
//
// Given the Block (or pixel) coordinates and the dimension of the texture
// in blocks (or pixels) this returns the twiddled offset of the block
// (or pixel) from the start of the map.
//
// NOTE the dimensions of the texture must be a power of 2
*/
/*****************************************************************************/

static int DisableTwiddlingRoutine = 0;

static U32 TwiddleUV(U32 YSize, U32 XSize, U32 YPos, U32 XPos)
{
	U32 Twiddled;

	U32 MinDimension;
	U32 MaxValue;

	U32 SrcBitPos;
	U32 DstBitPos;

	int ShiftCount;

	_ASSERT(YPos < YSize);
	_ASSERT(XPos < XSize);

	_ASSERT(POWER_OF_2(YSize));
	_ASSERT(POWER_OF_2(XSize));


	if(YSize < XSize)
	{
		MinDimension = YSize;
		MaxValue	 = XPos;
	}
	else
	{
		MinDimension = XSize;
		MaxValue	 = YPos;
	}

	/*
	// Nasty hack to disable twiddling
	*/
	if(DisableTwiddlingRoutine)
	{
		return (YPos* XSize + XPos);
	}

	/*
	// Step through all the bits in the "minimum" dimension
	*/
	SrcBitPos = 1;
	DstBitPos = 1;
	Twiddled  = 0;
	ShiftCount = 0;

	while(SrcBitPos < MinDimension)
	{
		if(YPos & SrcBitPos)
		{
			Twiddled |= DstBitPos;
		}

		if(XPos & SrcBitPos)
		{
			Twiddled |= (DstBitPos << 1);
		}


		SrcBitPos <<= 1;
		DstBitPos <<= 2;
		ShiftCount += 1;

	}/*end while*/

	/*
	// prepend any unused bits
	*/
	MaxValue >>= ShiftCount;

	Twiddled |=  (MaxValue << (2*ShiftCount));

	return Twiddled;
}

/***********************************************************/
/*
// Decompress
//
// Takes the compressed input data and outputs the equivalent decompressed
// image.
*/
/***********************************************************/

static void Decompress(AMTC_BLOCK_STRUCT *pCompressedData,
				const int Do2bitMode,
				const int XDim,
				const int YDim,
				const int AssumeImageTiles,
				unsigned char* pResultImage)
{
	int x, y;
	int i, j;

	int BlkX, BlkY;
	int BlkXp1, BlkYp1;
	int XBlockSize;
	int BlkXDim, BlkYDim;

	int StartX, StartY;

	int ModulationVals[8][16];
	int ModulationModes[8][16];

	int Mod, DoPT;

	unsigned int uPosition;

	/*
	// local neighbourhood of blocks
	*/
	AMTC_BLOCK_STRUCT *pBlocks[2][2];

	AMTC_BLOCK_STRUCT *pPrevious[2][2] = {{NULL, NULL}, {NULL, NULL}};

	/*
	// Low precision colours extracted from the blocks
	*/
	struct
	{
		int Reps[2][4];
	}Colours5554[2][2];

	/*
	// Interpolated A and B colours for the pixel
	*/
	int ASig[4], BSig[4];

	int Result[4];

	if(Do2bitMode)
	{
		XBlockSize = BLK_X_2BPP;
	}
	else
	{
		XBlockSize = BLK_X_4BPP;
	}


	/*
	// For MBX don't allow the sizes to get too small
	*/
	BlkXDim = _MAX(2, XDim / XBlockSize);
	BlkYDim = _MAX(2, YDim / BLK_Y_SIZE);

	/*
	// Step through the pixels of the image decompressing each one in turn
	//
	// Note that this is a hideously inefficient way to do this!
	*/
	for(y = 0; y < YDim; y++)
	{
		for(x = 0; x < XDim; x++)
		{
			/*
			// map this pixel to the top left neighbourhood of blocks
			*/
			BlkX = (x - XBlockSize/2);
			BlkY = (y - BLK_Y_SIZE/2);

			BlkX = LIMIT_COORD(BlkX, XDim, AssumeImageTiles);
			BlkY = LIMIT_COORD(BlkY, YDim, AssumeImageTiles);


			BlkX /= XBlockSize;
			BlkY /= BLK_Y_SIZE;

			//BlkX = LIMIT_COORD(BlkX, BlkXDim, AssumeImageTiles);
			//BlkY = LIMIT_COORD(BlkY, BlkYDim, AssumeImageTiles);


			/*
			// compute the positions of the other 3 blocks
			*/
			BlkXp1 = LIMIT_COORD(BlkX+1, BlkXDim, AssumeImageTiles);
			BlkYp1 = LIMIT_COORD(BlkY+1, BlkYDim, AssumeImageTiles);

			/*
			// Map to block memory locations
			*/
			pBlocks[0][0] = pCompressedData +TwiddleUV(BlkYDim, BlkXDim, BlkY, BlkX);
			pBlocks[0][1] = pCompressedData +TwiddleUV(BlkYDim, BlkXDim, BlkY, BlkXp1);
			pBlocks[1][0] = pCompressedData +TwiddleUV(BlkYDim, BlkXDim, BlkYp1, BlkX);
			pBlocks[1][1] = pCompressedData +TwiddleUV(BlkYDim, BlkXDim, BlkYp1, BlkXp1);


			/*
			// extract the colours and the modulation information IF the previous values
			// have changed.
			*/
			if(memcmp(pPrevious, pBlocks, 4*sizeof(void*)) != 0)
			{
				StartY = 0;
				for(i = 0; i < 2; i++)
				{
					StartX = 0;
					for(j = 0; j < 2; j++)
					{
						Unpack5554Colour(pBlocks[i][j], Colours5554[i][j].Reps);

						UnpackModulations(pBlocks[i][j],
										  Do2bitMode,
										  ModulationVals,
										  ModulationModes,
										  StartX, StartY);

						StartX += XBlockSize;
					}/*end for j*/

					StartY += BLK_Y_SIZE;
				}/*end for i*/

				/*
				// make a copy of the new pointers
				*/
				memcpy(pPrevious, pBlocks, 4*sizeof(void*));
			}/*end if the blocks have changed*/


			/*
			// decompress the pixel.  First compute the interpolated A and B signals
			*/
			InterpolateColours(Colours5554[0][0].Reps[0],
							   Colours5554[0][1].Reps[0],
							   Colours5554[1][0].Reps[0],
							   Colours5554[1][1].Reps[0],
							   Do2bitMode, x, y,
							   ASig);

			InterpolateColours(Colours5554[0][0].Reps[1],
							   Colours5554[0][1].Reps[1],
							   Colours5554[1][0].Reps[1],
							   Colours5554[1][1].Reps[1],
							   Do2bitMode, x, y,
							   BSig);

			GetModulationValue(x,y, Do2bitMode, (const int (*)[16])ModulationVals, (const int (*)[16])ModulationModes,
							   &Mod, &DoPT);

			/*
			// compute the modulated colour
			*/
			for(i = 0; i < 4; i++)
			{
				Result[i] = ASig[i] * 8 + Mod * (BSig[i] - ASig[i]);
				Result[i] >>= 3;
			}
			if(DoPT)
			{
				Result[3] = 0;
			}

			/*
			// Store the result in the output image
			*/
			uPosition = (x+y*XDim)<<2;
			pResultImage[uPosition+0] = (U8)Result[0];
			pResultImage[uPosition+1] = (U8)Result[1];
			pResultImage[uPosition+2] = (U8)Result[2];
			pResultImage[uPosition+3] = (U8)Result[3];

		}/*end for x*/
	}/*end for y*/

}

//---------------------------	ETC Compression ---------------------------------

#define _CLAMP_(X,Xmin,Xmax) (  (X)<(Xmax) ?  (  (X)<(Xmin)?(Xmin):(X)  )  : (Xmax)    )


unsigned int ETC_FLIP =  0x01000000;
unsigned int ETC_DIFF = 0x02000000;
const int mod[8][4]={{2, 8,-2,-8},
					{5, 17, -5, -17},
					{9, 29, -9, -29},
					{13, 42, -13, -42},
					{18, 60, -18, -60},
					{24, 80, -24, -80},
					{33, 106, -33, -106},
					{47, 183, -47, -183}};


// lsb: hgfedcba ponmlkji msb: hgfedcba ponmlkji due to endianness
unsigned long modifyPixel(int red, int green, int blue, int x, int y, unsigned long modBlock, int modTable)
{
	int index = x*4+y, pixelMod;
	unsigned long mostSig = modBlock<<1;
	if (index<8)	//hgfedcba
		pixelMod = mod[modTable][((modBlock>>(index+24))&0x1)+((mostSig>>(index+8))&0x2)];
	else	// ponmlkj
		pixelMod = mod[modTable][((modBlock>>(index+8))&0x1)+((mostSig>>(index-8))&0x2)];

	red = _CLAMP_(red+pixelMod,0,255);
	green = _CLAMP_(green+pixelMod,0,255);
	blue = _CLAMP_(blue+pixelMod,0,255);

	return ((red<<16) + (green<<8) + blue)|0xff000000;
}


// 
// returns Number of bytes of ETC data decompressed
//
// Description    : Convert ETC textures to RGBA8888. 
//					- pSrcData points to the source DXT data
//					- x and y are the dimensions of the source data, in pixels.
//					- pDestData is a pointer that will contain the output data.
//					- nMode indicates the format of the data (DXT 1 to 5)
//					
int ETCTextureDecompress(const void * const pSrcData, const int &x, const int &y, const void *pDestData,const int &nMode)
{
	unsigned long blockTop, blockBot, *input = (unsigned long*)pSrcData, *output;
	//unsigned long alphaTop, alphaBot, alphaValue[8];
	unsigned char red1, green1, blue1, red2, green2, blue2;
	bool bFlip, bDiff;
	int modtable1,modtable2;

	for(int i=0;i<y;i+=4)
	{
		for(int m=0;m<x;m+=4)
		{
				blockTop = *(input++);
				blockBot = *(input++);

			output = (unsigned long*)pDestData + i*x +m;
				// check flipbit
			bFlip = (blockTop & ETC_FLIP) != 0;
			bDiff = (blockTop & ETC_DIFF) != 0;

			if(bDiff)
			{	// differential mode 5 colour bits + 3 difference bits
				// get base colour for subblock 1
				blue1 = (unsigned char)((blockTop&0xf80000)>>16);
				green1 = (unsigned char)((blockTop&0xf800)>>8);
				red1 = (unsigned char)(blockTop&0xf8);

				// get differential colour for subblock 2
				signed char blues = (signed char)(blue1>>3) + ((signed char) ((blockTop & 0x70000) >> 11)>>5);
				signed char greens = (signed char)(green1>>3) + ((signed char)((blockTop & 0x700) >>3)>>5);
				signed char reds = (signed char)(red1>>3) + ((signed char)((blockTop & 0x7)<<5)>>5);

				blue2 = (unsigned char)blues;
				green2 = (unsigned char)greens;
				red2 = (unsigned char)reds;

				red1 = red1 +(red1>>5);	// copy bits to lower sig
				green1 = green1 + (green1>>5);	// copy bits to lower sig
				blue1 = blue1 + (blue1>>5);	// copy bits to lower sig

				red2 = (red2<<3) +(red2>>2);	// copy bits to lower sig
				green2 = (green2<<3) + (green2>>2);	// copy bits to lower sig
				blue2 = (blue2<<3) + (blue2>>2);	// copy bits to lower sig
			}
			else
			{	// individual mode 4 + 4 colour bits
				// get base colour for subblock 1
				blue1 = (unsigned char)((blockTop&0xf00000)>>16);
				blue1 = blue1 +(blue1>>4);	// copy bits to lower sig
				green1 = (unsigned char)((blockTop&0xf000)>>8);
				green1 = green1 + (green1>>4);	// copy bits to lower sig
				red1 = (unsigned char)(blockTop&0xf0);
				red1 = red1 + (red1>>4);	// copy bits to lower sig

				// get base colour for subblock 2
				blue2 = (unsigned char)((blockTop&0xf0000)>>12);
				blue2 = blue2 +(blue2>>4);	// copy bits to lower sig
				green2 = (unsigned char)((blockTop&0xf00)>>4);
				green2 = green2 + (green2>>4);	// copy bits to lower sig
				red2 = (unsigned char)((blockTop&0xf)<<4);
				red2 = red2 + (red2>>4);	// copy bits to lower sig
			}
			// get the modtables for each subblock
			modtable1 = (blockTop>>29)&0x7; 
			modtable2 = (blockTop>>26)&0x7; 

			if(!bFlip)
			{	// 2 2x4 blocks side by side

					for(int j=0;j<4;j++)	// vertical
					{
						for(int k=0;k<2;k++)	// horizontal
						{
							*(output+j*x+k) = modifyPixel(red1,green1,blue1,k,j,blockBot,modtable1);
							*(output+j*x+k+2) = modifyPixel(red2,green2,blue2,k+2,j,blockBot,modtable2);
						}
					}

			}
			else
			{	// 2 4x2 blocks on top of each other
					for(int j=0;j<2;j++)
					{
						for(int k=0;k<4;k++)
						{
							*(output+j*x+k) = modifyPixel(red1,green1,blue1,k,j,blockBot,modtable1);
							*(output+(j+2)*x+k) = modifyPixel(red2,green2,blue2,k,j+2,blockBot,modtable2);
						}
					}
			}
		}
	}
	return x*y/2;
}

int ETCDecompress(const void * const pSrcData,
						 const unsigned int &x,
						 const unsigned int &y,
						 void *pDestData,
						 const int &nMode)
{
	int i32read;
	if(x<ETC_MIN_TEXWIDTH || y<ETC_MIN_TEXHEIGHT)
	{	// decompress into a buffer big enought to take the minimum size
		char* pTempBuffer =	(char*) malloc(_MAX(x,ETC_MIN_TEXWIDTH)*_MAX(y,ETC_MIN_TEXHEIGHT)*4 * sizeof(char));
		memset(pTempBuffer, 0, _MAX(x,ETC_MIN_TEXWIDTH)*_MAX(y,ETC_MIN_TEXHEIGHT)*4 * sizeof(char));
		
		i32read = ETCTextureDecompress(pSrcData,_MAX(x,ETC_MIN_TEXWIDTH),_MAX(y,ETC_MIN_TEXHEIGHT),pTempBuffer,nMode);
		for(unsigned int i=0;i<y;i++)
		{	// copy from larger temp buffer to output data
			memcpy((char*)(pDestData)+i*x*4,pTempBuffer+_MAX(x,ETC_MIN_TEXWIDTH)*4*i,x*4);
		}
		if(pTempBuffer) 
		   free(pTempBuffer);
	}
	else	// decompress larger MIP levels straight into the output data
		i32read = ETCTextureDecompress(pSrcData,x,y,pDestData,nMode);

	// swap r and b channels
	unsigned char* pSwap=(unsigned char*)pDestData, swap;
	for(unsigned int i=0;i<y;i++)
		for(unsigned int j=0;j<x;j++)
		{
			swap = pSwap[0];
			pSwap[0] = pSwap[2];
			pSwap[2] = swap;
			pSwap+=4;
		}
	return i32read;
}
