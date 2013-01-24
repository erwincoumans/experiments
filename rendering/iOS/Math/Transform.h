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
#ifndef TRANSFORM_H_
#define TRANSFORM_H_

#include "Vector.h"
#include "Matrix.h"


/*!***************************************************************************
 @Function Name		TransVec3TransformArray
 @Output			pOut				Destination for transformed vectors
 @Input				nOutStride			Stride between vectors in pOut array
 @Input				pV					Input vector array
 @Input				nInStride			Stride between vectors in pV array
 @Input				pMatrix				Matrix to transform the vectors
 @Input				nNumberOfVertices	Number of vectors to transform
 @Description		Transform all vertices [X Y Z 1] in pV by pMatrix and
 					store them in pOut.
*****************************************************************************/
void TransVec3TransformArray(
	VECTOR4			* const pOut,
	const int			nOutStride,
	const VECTOR3	* const pV,
	const int			nInStride,
	const MATRIX	* const pMatrix,
	const int			nNumberOfVertices);

/*!***************************************************************************
 @Function			TransTransformArray
 @Output			pTransformedVertex	Destination for transformed vectors
 @Input				pV					Input vector array
 @Input				nNumberOfVertices	Number of vectors to transform
 @Input				pMatrix				Matrix to transform the vectors
 @Input				fW					W coordinate of input vector (e.g. use 1 for position, 0 for normal)
 @Description		Transform all vertices in pVertex by pMatrix and store them in
					pTransformedVertex
					- pTransformedVertex is the pointer that will receive transformed vertices.
					- pVertex is the pointer to untransformed object vertices.
					- nNumberOfVertices is the number of vertices of the object.
					- pMatrix is the matrix used to transform the object.
*****************************************************************************/
void TransTransformArray(
	VECTOR3			* const pTransformedVertex,
	const VECTOR3	* const pV,
	const int			nNumberOfVertices,
	const MATRIX	* const pMatrix,
	const float		fW = f2vt(1.0f));

/*!***************************************************************************
 @Function			TransTransformArrayBack
 @Output			pTransformedVertex
 @Input				pVertex
 @Input				nNumberOfVertices
 @Input				pMatrix
 @Description		Transform all vertices in pVertex by the inverse of pMatrix
					and store them in pTransformedVertex.
					- pTransformedVertex is the pointer that will receive transformed vertices.
					- pVertex is the pointer to untransformed object vertices.
					- nNumberOfVertices is the number of vertices of the object.
					- pMatrix is the matrix used to transform the object.
*****************************************************************************/
void TransTransformArrayBack(
	VECTOR3			* const pTransformedVertex,
	const VECTOR3	* const pVertex,
	const int			nNumberOfVertices,
	const MATRIX	* const pMatrix);

/*!***************************************************************************
 @Function			TransTransformBack
 @Output			pOut
 @Input				pV
 @Input				pM
 @Description		Transform vertex pV by the inverse of pMatrix
					and store in pOut.
*****************************************************************************/
void TransTransformBack(
	VECTOR4			* const pOut,
	const VECTOR4	* const pV,
	const MATRIX	* const pM);

/*!***************************************************************************
 @Function			TransTransform
 @Output			pOut
 @Input				pV
 @Input				pM
 @Description		Transform vertex pV by pMatrix and store in pOut.
*****************************************************************************/
void TransTransform(
	VECTOR4			* const pOut,
	const VECTOR4	* const pV,
	const MATRIX	* const pM);


#endif