/******************************************************************************

 @File         PVRTResourceFile.h

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
#ifndef RESOURCEFILE_H_
#define RESOURCEFILE_H_

#include <stdlib.h>
#include <string>
using namespace std;


/*!***************************************************************************
 @Class CPVRTResourceFile
 @Brief Simple resource file wrapper
*****************************************************************************/
class CPVRTResourceFile
{
public:
	/*!***************************************************************************
	@Function			SetReadPath
	@Input				pszReadPath The path where you would like to read from
	@Description		Sets the read path
	*****************************************************************************/
	static void SetReadPath(const char* pszReadPath);

	/*!***************************************************************************
	@Function			GetReadPath
	@Returns			The currently set read path
	@Description		Returns the currently set read path
	*****************************************************************************/
	static string GetReadPath();

	/*!***************************************************************************
	@Function			CPVRTResourceFile
	@Input				pszFilename Name of the file you would like to open
	@Description		Constructor
	*****************************************************************************/
	CPVRTResourceFile(const char* pszFilename);

	/*!***************************************************************************
	@Function			~CPVRTResourceFile
	@Description		Destructor
	*****************************************************************************/
	virtual ~CPVRTResourceFile();

	/*!***************************************************************************
	@Function			IsOpen
	@Returns			true if the file is open
	@Description		Is the file open
	*****************************************************************************/
	bool IsOpen() const;

	/*!***************************************************************************
	@Function			IsMemoryFile
	@Returns			true if the file was opened from memory
	@Description		Was the file opened from memory
	*****************************************************************************/
	bool IsMemoryFile() const;

	/*!***************************************************************************
	@Function			Size
	@Returns			The size of the opened file
	@Description		Returns the size of the opened file
	*****************************************************************************/
	size_t Size() const;

	/*!***************************************************************************
	@Function			DataPtr
	@Returns			A pointer to the file data
	@Description		Returns a pointer to the file data
	*****************************************************************************/
	const void* DataPtr() const;
	
	/*!***************************************************************************
	@Function			StringPtr
	@Returns			The file data as a string
	@Description		Returns the file as a null-terminated string
	*****************************************************************************/
	// convenience getter. Also makes it clear that you get a null-terminated buffer.
	const char* StringPtr() const;

	/*!***************************************************************************
	@Function			Close
	@Description		Closes the file
	*****************************************************************************/
	void Close();

protected:
	bool m_bOpen;
	bool m_bMemoryFile;
	size_t m_Size;
	const char* m_pData;

	static string s_ReadPath;
};

#endif // _PVRTRESOURCEFILE_H_

/*****************************************************************************
 End of file (PVRTResourceFile.h)
*****************************************************************************/
