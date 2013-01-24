/******************************************************************************

 @File         PVRTMemoryFileSystem.h

 @Title        

 @Copyright    Copyright (C) 2007 - 2008 by Imagination Technologies Limited.

 @Platform     ANSI compatible

 @Description  Memory file system for resource files

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
#ifndef MEMORYFILE_H_
#define MEMORYFILE_H_

#include <stddef.h>

class CPVRTMemoryFileSystem
{
public:	
	CPVRTMemoryFileSystem(const char* pszFilename, const void* pBuffer, size_t Size, bool bCopy = false);
	
	/*!***************************************************************************
	 @Function		RegisterMemoryFile
	 @Input			pszFilename		Name of file to register
	 @Input			pBuffer			Pointer to file data
	 @Input			Size			File size
	 @Input			bCopy			Name and data should be copied?
	 @Description	Registers a block of memory as a file that can be looked up
	                by name.
	*****************************************************************************/
	static void RegisterMemoryFile(const char* pszFilename, const void* pBuffer, size_t Size, bool bCopy = false);
	
	/*!***************************************************************************
	 @Function		GetFile
	 @Input			pszFilename		Name of file to open
	 @Output		ppBuffer		Pointer to file data
	 @Output		pSize			File size
	 @Return		true if the file was found in memory, false otherwise
	 @Description	Looks up a file in the memory file system by name. Returns a
	                pointer to the file data as well as its size on success.
	*****************************************************************************/
	static bool GetFile(const char* pszFilename, const void** ppBuffer, size_t* pSize);

	//static std::string DebugOut();

	/*!***************************************************************************
	 @Function		GetNumFiles
	 @Return		The number of registered files
	 @Description	Getter for the number of registered files
	*****************************************************************************/
	static int GetNumFiles();

	/*!***************************************************************************
	 @Function		GetFilename
	 @Input			i32Index		Index of file
	 @Return		A pointer to the filename of the requested file
	 @Description	Looks up a file in the memory file system by name. Returns a
	                pointer to the file data as well as its size on success.
	*****************************************************************************/
	static const char* GetFilename(int i32Index);
	
protected:	
	class CAtExit
	{
	public:
		/*!***************************************************************************
		@Function		Destructor
		@Description	Destructor of CAtExit class. Workaround for platforms that
		                don't support the atexit() function. This deletes any memory
						file system data.
		*****************************************************************************/
		~CAtExit();
	};
	static CAtExit s_AtExit;

	friend class CAtExit;
	
	struct SFileInfo
	{
		const char* pszFilename;
		const void* pBuffer;
		size_t Size;
		bool bAllocated;
	};
	static SFileInfo* s_pFileInfo;
	static int s_i32NumFiles;
	static int s_i32Capacity;
};

#endif // MEMORYFILE_H_