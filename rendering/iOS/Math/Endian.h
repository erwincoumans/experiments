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

#ifndef ENDIAN_H_
#define ENDIAN_H_


/****************************************************************************
 ** swap template function
 ****************************************************************************/
/*!***************************************************************************
 @Function		PVRTswap
 @Input			a Type a
 @Input			b Type b
 @Description	A swap template function that swaps a and b
 *****************************************************************************/

template <typename T>
inline void PVRTswap(T& a, T& b)
{
	T temp = a;
	a = b;
	b = temp;
}

#ifdef _UITRON_
template void PVRTswap<unsigned char>(unsigned char& a, unsigned char& b);
#endif

/*!***************************************************************************
 @Function		PVRTByteSwap
 @Input			pBytes A number
 @Input			i32ByteNo Number of bytes in pBytes
 @Description	Swaps the endianness of pBytes in place
 *****************************************************************************/
inline void PVRTByteSwap(unsigned char* pBytes, int i32ByteNo)
{
	int i = 0, j = i32ByteNo - 1;
	
	while(i < j)
		PVRTswap<unsigned char>(pBytes[i++], pBytes[j--]);
}

/*!***************************************************************************
 @Function		PVRTByteSwap32
 @Input			ui32Long A number
 @Returns		ui32Long with its endianness changed
 @Description	Converts the endianness of an unsigned long
 *****************************************************************************/
inline unsigned long PVRTByteSwap32(unsigned long ui32Long)
{
	return ((ui32Long&0x000000FF)<<24) + ((ui32Long&0x0000FF00)<<8) + ((ui32Long&0x00FF0000)>>8) + ((ui32Long&0xFF000000) >> 24);
}

/*!***************************************************************************
 @Function		PVRTByteSwap16
 @Input			ui16Short A number
 @Returns		ui16Short with its endianness changed
 @Description	Converts the endianness of a unsigned short
 *****************************************************************************/
inline unsigned short PVRTByteSwap16(unsigned short ui16Short)
{
	return (ui16Short>>8) | (ui16Short<<8);
}

/*!***************************************************************************
 @Function		PVRTIsLittleEndian
 @Returns		True if the platform the code is ran on is little endian
 @Description	Returns true if the platform the code is ran on is little endian
 *****************************************************************************/
inline bool PVRTIsLittleEndian()
{
	static bool bLittleEndian;
	static bool bIsInit = false;
	
	if(!bIsInit)
	{
		short int word = 0x0001;
		char *byte = (char*) &word;
		bLittleEndian = byte[0] ? true : false;
		bIsInit = true;
	}
	
	return bLittleEndian;
}


#endif // ENDIAN_H_

