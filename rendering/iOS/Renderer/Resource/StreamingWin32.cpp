/******************************************************************************

 @File         PVRTModelPOD.cpp

 @Title        

 @Copyright    Copyright (C) 2003 - 2008 by Imagination Technologies Limited.

 @Platform     ANSI compatible

 @Description  Code to load POD files - models exported from MAX.

******************************************************************************/
/*
All changes:
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

#include "StreamingWin32.h"

bool CSourceResource::Init(const TCHAR * const pszName)
{
	HRSRC	hR;
	HGLOBAL	hG;

	// Find the resource
	hR = FindResource(GetModuleHandle(NULL), pszName, RT_RCDATA);
	if(!hR)
		return false;

	// How big is the resource?
	m_nSize = SizeofResource(NULL, hR);
	if(!m_nSize)
		return false;

	// Get a pointer to the resource data
	hG = LoadResource(NULL, hR);
	if(!hG)
		return false;

	m_pData = (unsigned char*)LockResource(hG);
	if(!m_pData)
		return false;

	m_nReadPos = 0;
	return true;
}

bool CSourceResource::Read(void* lpBuffer, const unsigned int dwNumberOfBytesToRead)
{
	if(m_nReadPos + dwNumberOfBytesToRead > m_nSize)
		return false;

	_ASSERT(lpBuffer);
	memcpy(lpBuffer, &m_pData[m_nReadPos], dwNumberOfBytesToRead);
	m_nReadPos += dwNumberOfBytesToRead;
	return true;
}

bool CSourceResource::Skip(const unsigned int nBytes)
{
	if(m_nReadPos + nBytes > m_nSize)
		return false;

	m_nReadPos += nBytes;
	return true;
}
