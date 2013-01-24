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
#include "ResourceFile.h"
#include "MemoryManager.h"

/*!***************************************************************************
 Class: CSource
*****************************************************************************/
class CSource
{
public:
	/*!***************************************************************************
	@Function			~CSource
	@Description		Destructor
	*****************************************************************************/
	virtual ~CSource() {};
	virtual bool Read(void* lpBuffer, const unsigned int dwNumberOfBytesToRead) = 0;
	virtual bool Skip(const unsigned int nBytes) = 0;
	
	template <typename T>
	bool Read(T &n)
	{
		return Read(&n, sizeof(T));
	}

	bool ReadMarker(unsigned int &nName, unsigned int &nLen);

	template <typename T>
	bool ReadAfterAlloc(T* &lpBuffer, const unsigned int dwNumberOfBytesToRead)
	{
		if(!SafeAlloc(lpBuffer, dwNumberOfBytesToRead))
			return false;
		return Read(lpBuffer, dwNumberOfBytesToRead);
	}
};


/*!***************************************************************************
 Class: CSourceStream
*****************************************************************************/
class CSourceStream : public CSource
{
protected:
	CPVRTResourceFile* m_pFile;
	size_t m_BytesReadCount;

public:
	/*!***************************************************************************
	@Function			CSourceStream
	@Description		Constructor
	*****************************************************************************/
	CSourceStream() : m_pFile(0), m_BytesReadCount(0) {}

	/*!***************************************************************************
	@Function			~CSourceStream
	@Description		Destructor
	*****************************************************************************/
	virtual ~CSourceStream();

	bool Init(const char * const pszFileName);

	virtual bool Read(void* lpBuffer, const unsigned int dwNumberOfBytesToRead);
	virtual bool Skip(const unsigned int nBytes);
};

