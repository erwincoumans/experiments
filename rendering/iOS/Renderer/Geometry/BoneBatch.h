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
#ifndef _BONEBATCH_H_
#define _BONEBATCH_H_

#include "Vertex.h"

/*!***************************************************************************
 Handles a batch of bones
*****************************************************************************/
class CPVRTBoneBatches
{
public:
	int	*pnBatches;			/*!< Space for nBatchBoneMax bone indices, per batch */
	int	*pnBatchBoneCnt;	/*!< Actual number of bone indices, per batch */
	int	*pnBatchOffset;		/*!< Offset into triangle array, per batch */
	int nBatchBoneMax;		/*!< Stored value as was passed into Create() */
	int	nBatchCnt;			/*!< Number of batches to render */

public:
	/*!***********************************************************************
	 @Function		Create
	 @Output		pnVtxNumOut		vertex count
	 @Output		pVtxOut			Output vertices (program must free() this)
	 @Modified		pwIdx			index array for triangle list
	 @Input			nVtxNum			vertex count
	 @Input			pVtx			vertices
	 @Input			nStride			Size of a vertex (in bytes)
	 @Input			nOffsetWeight	Offset in bytes to the vertex bone-weights
	 @Input			eTypeWeight		Data type of the vertex bone-weights
	 @Input			nOffsetIdx		Offset in bytes to the vertex bone-indices
	 @Input			eTypeIdx		Data type of the vertex bone-indices
	 @Input			nTriNum			Number of triangles
	 @Input			nBatchBoneMax	Number of bones a batch can reference
	 @Input			nVertexBones	Number of bones affecting each vertex
	 @Returns		true if successful
	 @Description	Fills the bone batch structure
	*************************************************************************/
	bool Create(
		int					* const pnVtxNumOut,
		char				** const pVtxOut,
		unsigned short		* const pwIdx,
		const int			nVtxNum,
		const char			* const pVtx,
		const int			nStride,
		const int			nOffsetWeight,
		const EPVRTDataType	eTypeWeight,
		const int			nOffsetIdx,
		const EPVRTDataType	eTypeIdx,
		const int			nTriNum,
		const int			nBatchBoneMax,
		const int			nVertexBones);

	/*!***********************************************************************
	 @Function		Release
	 @Description	Destroy the bone batch structure
	*************************************************************************/
	void Release()
	{
		{ delete pnBatches;			pnBatches = 0; }
		{ delete pnBatchBoneCnt;	pnBatchBoneCnt = 0; }
		{ delete pnBatchOffset;		pnBatchOffset = 0; }
		nBatchCnt = 0;
	}
};


#endif /* _BONEBATCH_H_ */

/*****************************************************************************
 End of file (BoneBatch.h)
*****************************************************************************/
