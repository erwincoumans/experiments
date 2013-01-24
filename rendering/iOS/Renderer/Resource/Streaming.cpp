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

#include "Macros.h"

#include "Streaming.h"

bool CSource::ReadMarker(unsigned int &nName, unsigned int &nLen)
{
	if(!Read(&nName, sizeof(nName)))
		return false;
	if(!Read(&nLen, sizeof(nLen)))
		return false;
	return true;
}

/*!***************************************************************************
@Function			~CSourceStream
@Description		Destructor
*****************************************************************************/
CSourceStream::~CSourceStream()
{
	delete m_pFile;
}

bool CSourceStream::Init(const char * const pszFileName)
{
	m_BytesReadCount = 0;
	if (m_pFile) delete m_pFile;

	m_pFile = new CPVRTResourceFile(pszFileName);
	if (!m_pFile->IsOpen())
	{
		delete m_pFile;
		m_pFile = 0;
		return false;
	}
	return true;
}

bool CSourceStream::Read(void* lpBuffer, const unsigned int dwNumberOfBytesToRead)
{
	_ASSERT(lpBuffer);
	_ASSERT(m_pFile);

	if (m_BytesReadCount + dwNumberOfBytesToRead > m_pFile->Size()) return false;

	memcpy(lpBuffer, &(m_pFile->StringPtr())[m_BytesReadCount], dwNumberOfBytesToRead);

	m_BytesReadCount += dwNumberOfBytesToRead;
	return true;
}

bool CSourceStream::Skip(const unsigned int nBytes)
{
	if (m_BytesReadCount + nBytes > m_pFile->Size()) return false;
	m_BytesReadCount += nBytes;
	return true;
}

