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
#ifndef _VERTEX_H_
#define _VERTEX_H_

#include "Mathematics.h"

/****************************************************************************
** Enumerations
****************************************************************************/
enum EPVRTDataType 
{
	EPODDataNone,
	EPODDataFloat,
	EPODDataInt,
	EPODDataUnsignedShort,
	EPODDataRGBA,
	EPODDataARGB,
	EPODDataD3DCOLOR,
	EPODDataUBYTE4,
	EPODDataDEC3N,
	EPODDataFixed16_16,
	EPODDataUnsignedByte,
	EPODDataShort,
	EPODDataShortNorm,
	EPODDataByte,
	EPODDataByteNorm
};

/*!***************************************************************************
 @Function			DataTypeRead
 @Output			pV
 @Input				pData
 @Input				eType
 @Input				nCnt
 @Description		Read a vector
*****************************************************************************/
void PVRTVertexRead(
	VECTOR4		* const pV,
					const void			* const pData,
					const EPVRTDataType	eType,
					const int			nCnt);

/*!***************************************************************************
 @Function			PVRTVertexRead
 @Output			pV
 @Input				pData
 @Input				eType
 @Description		Read an int
*****************************************************************************/
void PVRTVertexRead(
					unsigned int		* const pV,
					const void			* const pData,
					const EPVRTDataType	eType);

/*!***************************************************************************
 @Function			PVRTVertexWrite
 @Output			pOut
 @Input				eType
 @Input				nCnt
 @Input				pV
 @Description		Write a vector
*****************************************************************************/
void PVRTVertexWrite(
					void				* const pOut,
					const EPVRTDataType	eType,
					const int			nCnt,
	const VECTOR4	* const pV);

/*!***************************************************************************
 @Function			PVRTVertexWrite
 @Output			pOut
 @Input				eType
 @Input				V
 @Description		Write an int
*****************************************************************************/
void PVRTVertexWrite(
					void				* const pOut,
					const EPVRTDataType	eType,
					const unsigned int	V);

/*!***************************************************************************
 @Function			VertexTangentBinormal
 @Output			pvTan
 @Output			pvBin
 @Input				pvNor
 @Input				pfPosA
 @Input				pfPosB
 @Input				pfPosC
 @Input				pfTexA
 @Input				pfTexB
 @Input				pfTexC
 @Description		Calculates the tangent and binormal vectors for
					vertex 'A' of the triangle defined by the 3 supplied
					3D position coordinates (pfPosX) and 2D texture
					coordinates (pfTexX).
*****************************************************************************/
void VertexTangentBinormal(
							VECTOR3			* const pvTan,
							VECTOR3			* const pvBin,
							const  VECTOR3	* const pvNor,
							const float		* const pfPosA,
							const float		* const pfPosB,
							const float		* const pfPosC,
							const float		* const pfTexA,
							const float		* const pfTexB,
							const float		* const pfTexC);

/*!***************************************************************************
 @Function			VertexGenerateTangentSpace
 @Output			pnVtxNumOut			Output vertex count
 @Output			pVtxOut				Output vertices (program must free() this)
 @Modified			pwIdx				input AND output; index array for triangle list
 @Input				nVtxNum				Input vertex count
 @Input				pVtx				Input vertices
 @Input				nStride				Size of a vertex (in bytes)
 @Input				nOffsetPos			Offset in bytes to the vertex position
 @Input				eTypePos			Data type of the position
 @Input				nOffsetNor			Offset in bytes to the vertex normal
 @Input				eTypeNor			Data type of the normal
 @Input				nOffsetTex			Offset in bytes to the vertex texture coordinate to use
 @Input				eTypeTex			Data type of the texture coordinate
 @Input				nOffsetTan			Offset in bytes to the vertex tangent
 @Input				eTypeTan			Data type of the tangent
 @Input				nOffsetBin			Offset in bytes to the vertex binormal
 @Input				eTypeBin			Data type of the binormal
 @Input				nTriNum				Number of triangles
 @Input				fSplitDifference	Split a vertex if the DP3 of tangents/binormals are below this (range -1..1)
 @Return			false if there was a problem.
 @Description		Calculates the tangent space for all supplied vertices.
					Writes tangent and binormal vectors to the output
					vertices, copies all other elements from input vertices.
					Will split vertices if necessary - i.e. if two triangles
					sharing a vertex want to assign it different
					tangent-space matrices. The decision whether to split
					uses fSplitDifference - of the DP3 of two desired
					tangents or two desired binormals is higher than this,
					the vertex will be split.
*****************************************************************************/
bool VertexGenerateTangentSpace(
								int				* const pnVtxNumOut,
								char			** const pVtxOut,
								unsigned short	* const pwIdx,
								const int		nVtxNum,
								const char		* const pVtx,
								const int		nStride,
								const int		nOffsetPos,
								EPVRTDataType		eTypePos,
								const int		nOffsetNor,
								EPVRTDataType		eTypeNor,
								const int		nOffsetTex,
								EPVRTDataType		eTypeTex,
								const int		nOffsetTan,
								EPVRTDataType		eTypeTan,
								const int		nOffsetBin,
								EPVRTDataType		eTypeBin,
								const int		nTriNum,
								const float		fSplitDifference);


#endif 