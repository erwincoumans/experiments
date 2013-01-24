/******************************************************************************

 @File         PVRTResourceFile.cpp

 @Title        

 @Copyright    Copyright (C) 2007 - 2008 by Imagination Technologies Limited.

 @Platform     ANSI compatible

 @Description  Simple resource file wrapper

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
#include <stdio.h>

#include "MemoryFile.h"

string CPVRTResourceFile::s_ReadPath;

/*!***************************************************************************
@Function			SetReadPath
@Input				pszReadPath The path where you would like to read from
@Description		Sets the read path
*****************************************************************************/
void CPVRTResourceFile::SetReadPath(const char* const pszReadPath)
{
	s_ReadPath = (pszReadPath) ? pszReadPath : "";
}

/*!***************************************************************************
@Function			GetReadPath
@Returns			The currently set read path
@Description		Returns the currently set read path
*****************************************************************************/
string CPVRTResourceFile::GetReadPath()
{
	return string(s_ReadPath);
}

/*!***************************************************************************
@Function			CPVRTResourceFile
@Input				pszFilename Name of the file you would like to open
@Description		Constructor
*****************************************************************************/
CPVRTResourceFile::CPVRTResourceFile(const char* const pszFilename) :
	m_bOpen(false),
	m_bMemoryFile(false),
	m_Size(0),	m_pData(0)
{
	string Path(s_ReadPath);
	Path += pszFilename;
	
	FILE* pFile = fopen(Path.c_str(), "rb");
	if (pFile)
	{
		// Get the file size
		fseek(pFile, 0, SEEK_END);
		m_Size = ftell(pFile);
		fseek(pFile, 0, SEEK_SET);

		// read the data, append a 0 byte as the data might represent a string
		char* pData = new char[m_Size + 1];
		pData[m_Size] = '\0';
		size_t BytesRead = fread(pData, 1, m_Size, pFile);

		if (BytesRead != m_Size)
		{
			delete [] pData;
			m_Size = 0;
		}
		else
		{
			m_pData = pData;
			m_bOpen = true;
		}
		fclose(pFile);
	}

	if (!m_bOpen)
	{
		m_bOpen = m_bMemoryFile = CPVRTMemoryFileSystem::GetFile(pszFilename, (const void**)(&m_pData), &m_Size);
	}
}

/*!***************************************************************************
@Function			~CPVRTResourceFile
@Description		Destructor
*****************************************************************************/
CPVRTResourceFile::~CPVRTResourceFile()
{
	Close();
}

/*!***************************************************************************
@Function			IsOpen
@Returns			true if the file is open
@Description		Is the file open
*****************************************************************************/
bool CPVRTResourceFile::IsOpen() const
{
	return m_bOpen;
}

/*!***************************************************************************
@Function			IsMemoryFile
@Returns			true if the file was opened from memory
@Description		Was the file opened from memory
*****************************************************************************/
bool CPVRTResourceFile::IsMemoryFile() const
{
	return m_bMemoryFile;
}

/*!***************************************************************************
@Function			Size
@Returns			The size of the opened file
@Description		Returns the size of the opened file
*****************************************************************************/
size_t CPVRTResourceFile::Size() const
{
	return m_Size;
}

/*!***************************************************************************
@Function			DataPtr
@Returns			A pointer to the file data
@Description		Returns a pointer to the file data
*****************************************************************************/
const void* CPVRTResourceFile::DataPtr() const
{
	return m_pData;
}

/*!***************************************************************************
@Function			StringPtr
@Returns			The file data as a string
@Description		Returns the file as a null-terminated string
*****************************************************************************/
const char* CPVRTResourceFile::StringPtr() const
{
	return m_pData;
}

/*!***************************************************************************
@Function			Close
@Description		Closes the file
*****************************************************************************/
void CPVRTResourceFile::Close()
{
	if (m_bOpen)
	{
		if (!m_bMemoryFile)
		{
			delete [] (char*)m_pData;
		}
		m_bMemoryFile = false;
		m_bOpen = false;
		m_pData = 0;
		m_Size = 0;
	}
}

//#if defined(BUILD_OGLES2) || defined(BUILD_OGLES) || defined(BUILD_OGL) || defined(BUILD_OVG) || defined(BUILD_DX9) || defined(BUILD_DX10)

/****************************************************************************
** class CPVRTMemoryFileSystem
****************************************************************************/
CPVRTMemoryFileSystem::CAtExit CPVRTMemoryFileSystem::s_AtExit;
CPVRTMemoryFileSystem::SFileInfo* CPVRTMemoryFileSystem::s_pFileInfo = 0;
int CPVRTMemoryFileSystem::s_i32Capacity = 0;
int CPVRTMemoryFileSystem::s_i32NumFiles = 0;

/*!***************************************************************************
@Function		Destructor
@Description	Destructor of CAtExit class. Workaround for platforms that
		        don't support the atexit() function. This deletes any memory
				file system data.
*****************************************************************************/
CPVRTMemoryFileSystem::CAtExit::~CAtExit()
{
	for (int i = 0; i < CPVRTMemoryFileSystem::s_i32NumFiles; ++i)
	{
		if (CPVRTMemoryFileSystem::s_pFileInfo[i].bAllocated)
		{
			delete [] (char*)CPVRTMemoryFileSystem::s_pFileInfo[i].pszFilename;
			delete [] (char*)CPVRTMemoryFileSystem::s_pFileInfo[i].pBuffer;
		}
	}
	delete [] CPVRTMemoryFileSystem::s_pFileInfo;
}

CPVRTMemoryFileSystem::CPVRTMemoryFileSystem(const char* pszFilename, const void* pBuffer, size_t Size, bool bCopy)
{
	RegisterMemoryFile(pszFilename, pBuffer, Size, bCopy);
}

/*!***************************************************************************
@Function		RegisterMemoryFile
@Input			pszFilename		Name of file to register
@Input			pBuffer			Pointer to file data
@Input			Size			File size
@Input			bCopy			Name and data should be copied?
@Description	Registers a block of memory as a file that can be looked up
				by name.
*****************************************************************************/
void CPVRTMemoryFileSystem::RegisterMemoryFile(const char* pszFilename, const void* pBuffer, size_t Size, bool bCopy)
{
	if (s_i32NumFiles == s_i32Capacity)
	{
		SFileInfo* pFileInfo = new SFileInfo[s_i32Capacity + 10];
		memcpy(pFileInfo, s_pFileInfo, sizeof(SFileInfo) * s_i32Capacity);
		delete [] s_pFileInfo;
		s_pFileInfo = pFileInfo;
		s_i32Capacity += 10;
	}

	s_pFileInfo[s_i32NumFiles].pszFilename = pszFilename;
	s_pFileInfo[s_i32NumFiles].pBuffer = pBuffer;
	if (bCopy)
	{
		char* pszNewFilename = new char[strlen(pszFilename)];
		strcpy(pszNewFilename, pszFilename);
		s_pFileInfo[s_i32NumFiles].pszFilename = pszNewFilename;

		void* pszNewBuffer = new char[Size];
		memcpy(pszNewBuffer, pBuffer, Size);
		s_pFileInfo[s_i32NumFiles].pBuffer = pszNewBuffer;
	}
	s_pFileInfo[s_i32NumFiles].Size = Size;
	s_pFileInfo[s_i32NumFiles].bAllocated = bCopy;
	++s_i32NumFiles;
}

/*!***************************************************************************
@Function		GetFile
@Input			pszFilename		Name of file to open
@Output			ppBuffer		Pointer to file data
@Output			pSize			File size
@Return			true if the file was found in memory, false otherwise
@Description	Looks up a file in the memory file system by name. Returns a
				pointer to the file data as well as its size on success.
*****************************************************************************/
bool CPVRTMemoryFileSystem::GetFile(const char* pszFilename, const void** ppBuffer, size_t* pSize)
{
	for (int i = 0; i < s_i32NumFiles; ++i)
	{
		if (strcmp(s_pFileInfo[i].pszFilename, pszFilename) == 0)
		{
			if (ppBuffer) *ppBuffer = s_pFileInfo[i].pBuffer;
			if (pSize) *pSize = s_pFileInfo[i].Size;
			return true;
		}
	}
	return false;
}

/*!***************************************************************************
@Function		GetNumFiles
@Return			The number of registered files
@Description	Getter for the number of registered files
*****************************************************************************/
int CPVRTMemoryFileSystem::GetNumFiles()
{
	return s_i32NumFiles;
}

/*!***************************************************************************
@Function		GetFilename
@Input			i32Index		Index of file
@Return			A pointer to the filename of the requested file
@Description	Looks up a file in the memory file system by name. Returns a
				pointer to the file data as well as its size on success.
*****************************************************************************/
const char* CPVRTMemoryFileSystem::GetFilename(int i32Index)
{
	if (i32Index < 0 || i32Index > s_i32NumFiles) return 0;

	return s_pFileInfo[i32Index].pszFilename;
}

//std::string CPVRTMemoryFileSystem::DebugOut()
//{
//	std::stringstream Out;
//
//	Out << "PVRTMemoryFileSystem Info: Files - " << s_i32NumFiles << "\n";
//	for (int i = 0; i < s_i32NumFiles; ++i)
//	{
//		Out << "\t" << i << " " << s_pFileInfo[i].pszFilename << " (" 
//			<< s_pFileInfo[i].Size << ", " << (int)s_pFileInfo[i].pBuffer << ")\n";
//	}
//
//	return Out.str();
//}

//#endif